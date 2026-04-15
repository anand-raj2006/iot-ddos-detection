const express = require('express');
const mongoose = require('mongoose');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

// 1. Setup Express App
const app = express();
app.use(cors());
app.use(express.json());

// 2. Connect to MongoDB Atlas using environment variable
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
    console.error('Missing MONGO_URI in environment variables.');
    process.exit(1);
}

mongoose.connect(MONGO_URI)
    .then(() => console.log("✅ Connected to MongoDB Atlas"))
    .catch((err) => console.log("❌ MongoDB Connection Error:", err));

// 3. Create Database Schema
const trafficSchema = new mongoose.Schema({
    flow_duration: Number,
    Header_Length: Number,
    "Protocol Type": Number, // Quotes needed due to space
    Rate: Number,
    Srate: Number,
    ack_count: Number,
    syn_count: Number,
    rst_count: Number,
    "Tot size": Number,      // Quotes needed due to space
    IAT: Number,
    prediction: String,      // Will hold "Attack" or "Normal"
    timestamp: { type: Date, default: Date.now }
});

const TrafficData = mongoose.model('TrafficData', trafficSchema);

// 4. POST Endpoint (Process Data, Ask Flask, Save to DB)
app.post('/data', async (req, res) => {
    try {
        const inputData = req.body;

        // Error Handling: Check for missing fields
        const requiredFields = [
            "flow_duration", "Header_Length", "Protocol Type", "Rate",
            "Srate", "ack_count", "syn_count", "rst_count", "Tot size", "IAT"
        ];

        for (let field of requiredFields) {
            if (inputData[field] === undefined) {
                return res.status(400).json({ error: `Missing required field: ${field}` });
            }
        }

        // Ask Flask for the ML Prediction
        let flaskResponse;
        try {
            flaskResponse = await axios.post('http://127.0.0.1:5000/predict', inputData);
        } catch (error) {
            return res.status(500).json({ error: "Flask ML API is down or unreachable." });
        }

        // Extract prediction (adjust based on your exact Flask output key)
        const predictionResult = flaskResponse.data.status || flaskResponse.data.prediction_label;

        // Save to MongoDB
        const newRecord = new TrafficData({
            ...inputData,
            prediction: predictionResult
        });
        await newRecord.save();

        // Send final response
        res.status(201).json({
            message: "Data processed and saved successfully!",
            prediction: predictionResult,
            data: newRecord
        });

    } catch (error) {
        res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
});

// 5. GET Endpoint (Fetch all history for Dashboard)
app.get('/data', async (req, res) => {
    try {
        const allData = await TrafficData.find().sort({ timestamp: -1 });
        res.status(200).json(allData);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch data from database" });
    }
});

// 6. Start the Server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`🚀 Node.js Backend running on http://localhost:${PORT}`);
});