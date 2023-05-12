import React, { useEffect, useRef, useState } from "react";

const Classifier = () => {
    const canvasRef = useRef();
    const imageRef = useRef();
    const videoRef = useRef();

    const [result, setResult] = useState("");

    useEffect(() => {
        // Event hook to fetch/get camera feed.
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
        // Event hook to send images to API.
        const interval = setInterval(async () => {
            captureImageFromCamera(); 

            if (imageRef.current) {
                const formData = new FormData();
                formData.append("image", imageRef.current);

                const response = await fetch("/classify", {
                    method: "POST",
                    body: formData,
                });
                alert(await response.text());
                if (response.status === 200) {
                    const textResponse = await response.text();
                    setResult(textResponse);
                } else {
                    setResult("Error from image API!");
                }
            }
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // Start video stream when element is loaded and ready.
    const playCameraFeed = () => {
        if (videoRef.current) {
            videoRef.current.play();
        }
    };

    const captureImageFromCamera = () => {
        const context = canvasRef.current.getContext("2d");
        const { videoWidth, videoHeight } = videoRef.current;

        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);

        canvasRef.current.toBlob((blob) => {
            imageRef.current = blob;
        })
    };

    return (
        <>
            <header>
                <h1>Image Classifier</h1>
            </header>
            <main>
                <video ref={videoRef} onCanPlay={() => playCameraFeed()} id="video" />
                <canvas ref={canvasRef} hidden></canvas>
                <p>This is: {result}</p>
            </main>
        </>
    )
};

export default Classifier;
