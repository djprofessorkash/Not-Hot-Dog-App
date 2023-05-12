import React, { useEffect, useRef, useState } from "react";

const Classifier = () => {
    const canvasRef = useRef();
    const imageRef = useRef();
    const videoRef = useRef();

    const [result, setResult] = useState("");

    useEffect(() => {
        // Fetch/get camera feed here.
        async function getCameraFeed() {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: true
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
        };

        getCameraFeed();
    }, []);

    useEffect(() => {
        // TODO: Send images to API here.
    }, []);

    // Start video stream when element is loaded and ready.
    const playCameraFeed = () => {
        if (videoRef.current) {
            videoRef.current.play();
        }
    };

    return (
        <>
            <header>
                <h1>Image Classifier</h1>
            </header>
            <main>
                <video ref={videoRef} onCanPlay={() => playCameraFeed()} id="video" />
            </main>
        </>
    )
};

export default Classifier;
