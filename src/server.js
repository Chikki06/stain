const express = require('express');
const formidable = require('formidable');
const path = require('path');
const fs = require('fs');
const http = require('http');
const socketIo = require('socket.io');
const session = require('express-session');
const winston = require('winston');
const { exec } = require('child_process');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);
const PORT = process.env.PORT || 3000;

// Add session middleware configuration
app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // set to true if using https
}));

// Path to the specific Python executable
const pythonExec = 'C:\\Users\\hiidk\\.conda\\envs\\research\\python.exe';


// Configure logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.json(),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'combined.log' })
    ]
});

// Ensure all required directories exist
const uploadDir = path.join(__dirname, '../files');
const bsqDir = path.join(uploadDir, 'BSQfiles');
const pngDir = path.join(uploadDir, 'pngs');
const outputPngDir = path.join(uploadDir, 'output-pngs');
const tiffDir = path.join(uploadDir, 'tiffs');

[uploadDir, bsqDir, pngDir, outputPngDir, tiffDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
        logger.info('Created directory:', dir);
    }
});

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, '../public')));

// Add this line to serve PNG files
app.use('/files', express.static(path.join(__dirname, '../files')));


// Handle folder upload
app.post('/upload', (req, res) => {
    logger.info('Folder upload started');
    const form = new formidable.IncomingForm({
        maxFileSize: 15 * 1024 * 1024 * 1024, // Increase the limit to 15GB
        maxTotalFileSize: 15 * 1024 * 1024 * 1024 // Increase the limit to 15GB
    });
    form.uploadDir = bsqDir; // Changed to BSQfiles directory
    form.keepExtensions = true; // Keep file extensions

    const uploadStartTime = Date.now(); // Start the upload timer

    form.parse(req, (err, fields, files) => {
        if (err) {
            logger.error('Error during form parsing:', err);
            res.status(500).json({ error: 'Error during upload' });
            return;
        }

        let uploadedFiles = files.upload;
        if (!Array.isArray(uploadedFiles)) {
            uploadedFiles = [uploadedFiles];
        }

        const bsqFile = uploadedFiles.find(file => path.extname(file.originalFilename) === '.bsq');
        const hdrFile = uploadedFiles.find(file => path.extname(file.originalFilename) === '.hdr');

        if (!bsqFile || !hdrFile) {
            logger.error('Both .bsq and .hdr files are required.');
            res.status(400).json({ error: 'Both .bsq and .hdr files are required.' });
            return;
        }

        // Ensure the destination directory exists with a unique name
        const timestamp = Date.now();
        req.session.timestamp = timestamp;
        const destDir = path.join(bsqDir, `${path.dirname(bsqFile.originalFilename)}_${timestamp}`);
        fs.mkdirSync(destDir, { recursive: true });

        // Move the files to the upload directory
        const bsqFilePath = path.join(destDir, path.basename(bsqFile.originalFilename));
        const hdrFilePath = path.join(destDir, path.basename(hdrFile.originalFilename));

        fs.renameSync(bsqFile.filepath, bsqFilePath);
        fs.renameSync(hdrFile.filepath, hdrFilePath);

        const uploadEndTime = Date.now(); // End the upload timer
        const uploadTime = (uploadEndTime - uploadStartTime) / 1000; // in seconds
        const totalSize = uploadedFiles.reduce((acc, file) => acc + file.size, 0);

        // Store upload time, size, and file paths in session
        req.session.uploadTime = uploadTime;
        req.session.uploadSize = totalSize;
        req.session.destDir = destDir;

        // Define the output directory
        const outputDir = path.join(__dirname, '../files/pngs');
        fs.mkdirSync(outputDir, { recursive: true });

        // Start PNG processing timer
        const pngStartTime = Date.now();

        // Run the Python script and capture the output path
        logger.info('Python script started');
        exec(`${pythonExec} ${path.join(__dirname, 'convert_bsq_to_png.py')} ${destDir} ${outputDir} ${timestamp}`, (error, stdout, stderr) => {
            // End PNG processing timer
            const pngEndTime = Date.now();
            const pngProcessTime = (pngEndTime - pngStartTime) / 1000; // in seconds

            if (error) {
                logger.error(`Error executing Python script: ${error.message}`);
                res.status(500).json({ error: 'Error generating image' });
                return;
            }

            const outputPath = stdout.trim();
            if (!outputPath) {
                logger.error('No output path returned from Python script.');
                res.status(500).json({ error: 'Error generating image' });
                return;
            }

            // Store the output path and PNG processing time in session
            req.session.outputPath = outputPath;
            req.session.pngProcessTime = pngProcessTime;

            logger.info('Processing completed successfully');
            const serverUrl = `${req.protocol}://${req.get('host')}`;
            const redirectUrl = `/success?uploadTime=${uploadTime}&uploadSize=${totalSize}&outputPath=${encodeURIComponent(outputPath)}&pngProcessTime=${pngProcessTime}&serverUrl=${encodeURIComponent(serverUrl)}`;
            res.redirect(redirectUrl);
        });
    });
});

// Add new endpoint for staining
app.post('/stain', express.json(), (req, res) => {
    const { coordinates } = req.body;
    const timestamp = req.session.timestamp;
    const destDir = req.session.destDir;
    const originalFolderName = path.basename(destDir).split('_')[0];
    
    // Use the original coordinates (0-1000 range)
    const bsqCoordinates = {
        x1: coordinates.x1,
        y1: coordinates.y1,
        x2: coordinates.x2,
        y2: coordinates.y2
    };

    const stainStartTime = Date.now();

    // Ensure both output directories exist
    const tiffsDir = path.join(__dirname, '../files/tiffs');
    const outputPngsDir = path.join(__dirname, '../files/output-pngs');
    fs.mkdirSync(tiffsDir, { recursive: true });
    fs.mkdirSync(outputPngsDir, { recursive: true });

    // Run Python script with coordinates
    const command = `${pythonExec} ${path.join(__dirname, 'testcrop.py')} ` +
        `--input_dir "${destDir}" ` +
        `--output_dir "${tiffsDir}" ` +
        `--coordinates ${bsqCoordinates.x1} ${bsqCoordinates.y1} ${bsqCoordinates.x2} ${bsqCoordinates.y2} ` +
        `--timestamp ${timestamp} ` +
        `--folder_name ${originalFolderName} ` +
        `--image_width ${coordinates.imageWidth} ` +
        `--image_height ${coordinates.imageHeight} ` +
        `--server_url ${req.protocol}://${req.get('host')}`;

    logger.info('Starting testcrop.py execution');
    console.log('Running command:', command);
    
    exec(command, (error, stdout, stderr) => {
        const stainTime = ((Date.now() - stainStartTime) / 1000).toFixed(2); // in seconds

        if (error) {
            logger.error(`Error executing testcrop.py: ${error.message}`);
            res.status(500).json({ error: 'Error processing image' });
            return;
        }

        // Parse output paths and times from Python script
        const outputs = stdout.trim().split('\n');
        const modelTime = parseFloat(outputs.find(line => line.startsWith('MODEL:'))?.split(':')[1] || '0').toFixed(2);
        const tiffPath = outputs.find(line => line.startsWith('TIFF:'))?.split(':')[1]?.trim();
        const tiffTime = parseFloat(outputs.find(line => line.startsWith('TIFF_TIME:'))?.split(':')[1] || '0').toFixed(2);
        const pngPath = outputs.find(line => line.startsWith('PNG:'))?.split(':')[1]?.trim();
        const pngTime = parseFloat(outputs.find(line => line.startsWith('PNG_TIME:'))?.split(':')[1] || '0').toFixed(2);

        logger.info('testcrop.py execution completed successfully');

        // Get just the filenames
        const tiffName = path.basename(tiffPath);
        const pngName = path.basename(pngPath);

        res.json({ 
            success: true, 
            redirect: `/output?png=${pngName}&tiff=${tiffName}&modelTime=${modelTime}&tiffTime=${tiffTime}&pngTime=${pngTime}&stainTime=${stainTime}&folderName=${originalFolderName}&timestamp=${timestamp}&x1=${coordinates.x1}&y1=${coordinates.y1}&x2=${coordinates.x2}&y2=${coordinates.y2}`
        });
    });
});

// Serve the output page
app.get('/output', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/output.html'));
});

// Handle file downloads
app.get('/download/:type/:filename', (req, res) => {
    const { type, filename } = req.params;
    
    // Sanitize filename to prevent directory traversal
    const sanitizedFilename = path.basename(filename);
    let filePath;
    
    if (type === 'png') {
        filePath = path.join(__dirname, '..', 'files', 'output-pngs', sanitizedFilename);
    } else if (type === 'tiff') {
        filePath = path.join(__dirname, '..', 'files', 'tiffs', sanitizedFilename);
    } else {
        return res.status(400).send('Invalid file type');
    }

    // Check if file exists before attempting download
    if (!fs.existsSync(filePath)) {
        return res.status(404).send('File not found');
    }

    res.download(filePath, sanitizedFilename, (err) => {
        if (err) {
            logger.error(`Error downloading file: ${err.message}`);
            // Only send error if headers haven't been sent yet
            if (!res.headersSent) {
                res.status(500).send('Error downloading file');
            }
        }
    });
});

// Serve the home page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Endpoint to get upload details from session
app.get('/upload-details', (req, res) => {
    res.json({
        time: req.session.uploadTime, // Transmitting upload time
        size: req.session.uploadSize, // Transmitting upload size
        outputPath: req.session.outputPath,
        pngProcessTime: req.session.pngProcessTime // Transmitting PNG processing time
    });
});

// Serve the success page
app.get('/success', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/success.html'));
});

// Start the server
server.listen(PORT, () => {
    logger.info(`Server is running on http://localhost:${PORT}`);
});