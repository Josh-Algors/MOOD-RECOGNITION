// // Load models

const videoElement = document.getElementById('video');
const startButton = document.querySelector('.btn.start')
const stopButton = document.querySelector('.btn.stop')
const canvas = document.getElementById('canvas');

canvas.style.display = 'none'

let blazeFaceModel;
let emotionModel;

function startVideo() {
    const videoElement = document.getElementById('video');

    return new Promise((resolve, reject) => {
        videoElement.addEventListener('loadeddata', () => {
            resolve(videoElement);
        });

        videoElement.addEventListener('error', (error) => {
            reject(error);
        });

        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                videoElement.srcObject = stream;
            })
            .catch((error) => {
                reject(error);
            });
    });
}

async function loadModels() {
    blazeFaceModel = await blazeface.load();
    emotionModel = await tf.loadLayersModel('./models/model.json');
}

function detectEmotions() {
    const videoElement = document.getElementById('video');
    //   const canvas = document.getElementById('canvas');
    canvas.style.display = 'block'
    canvas.style.position = 'absolute';

    canvas.style.top = '0';
    canvas.style.left = '0';

    const context = canvas.getContext('2d');

    setInterval(async () => {
        const predictions = await blazeFaceModel.estimateFaces(videoElement);
        context.clearRect(0, 0, canvas.width, canvas.height);

        predictions.forEach(async (prediction) => {
            const start = prediction.topLeft;
            const end = prediction.bottomRight;
            const size = [end[0] - start[0], end[1] - start[1]];

            // Draw face bounding box
            context.beginPath();
            context.lineWidth = '5';
            context.strokeStyle = 'red';
            context.rect(start[0], start[1], size[0], size[0]);
            context.stroke();

            // Crop and resize face region
            const faceCanvas = document.createElement('canvas');
            const faceContext = faceCanvas.getContext('2d');
            faceCanvas.width = 48;
            faceCanvas.height = 48;
            faceContext.drawImage(
                videoElement,
                start[0], start[1], size[0], size[1],
                0, 0, faceCanvas.width, faceCanvas.height
            );

            // Perform emotion recognition
            const emotions = ['Why are you angry? 😡', 'Why are you disgusted 🤮', 'Dont be fearful 😵‍💫', 'You look happy 😝', 'neutral look 😶', 'Dont be sad 😞', 'You look surprised 😳'];

            var imgData = faceContext.getImageData(0, 0, faceCanvas.width, faceCanvas.height);

            // Convert image data to grayscale
            const grayscaleData = new Uint8Array(faceCanvas.width * faceCanvas.height);
            for (let i = 0; i < imgData.data.length; i += 4) {
                const r = imgData.data[i];
                const g = imgData.data[i + 1];
                const b = imgData.data[i + 2];
                const grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                grayscaleData[i / 4] = grayscale;
            }

            // Create tensor from grayscale image data
            const imgTensor = tf.tensor4d(grayscaleData, [1, faceCanvas.height, faceCanvas.width, 1]);

            // Normalize the image
            const normalized = imgTensor.div(255.0);
            const predictions = await emotionModel.predict(normalized);
            const probabilities = predictions.dataSync();
            const maxProbability = Math.max(...probabilities);
            const maxIndex = probabilities.indexOf(maxProbability);
            const emotion = emotions[maxIndex];

            // Display emotion label
            context.fillStyle = 'red';
            context.font = '16px Arial';
            context.fillText(emotion, start[0], start[1] > 10 ? start[1] - 5 : 10);

            imgData = null;
            predictions.dispose();
        });
    }, 1000);
}

// function to stop web cam
function stop() {
    var stream = video.srcObject;
    var tracks = stream.getTracks();

    for (var i = 0; i < tracks.length; i++) {
        var track = tracks[i];
        track.stop();
    }

    video.srcObject = null;
    canvas.style.display = 'none'
}

startButton.addEventListener('click', () => {
    loadModels()
        .then(() => {
            return startVideo();
        })
        .then((videoElement) => {
            detectEmotions(videoElement);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
})


stopButton.addEventListener('click', () => {
    stop()
})

