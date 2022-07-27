geoUrlAPI = "https://json.geoiplookup.io/"

geoResponse = httpGet(geoUrlAPI);
const geoJSON = JSON.parse(geoResponse);
console.log(geoJSON);

const approxCity = geoJSON.city;
console.log(approxCity);

const approxLat =  geoJSON.latitude;
console.log(approxLat);

const approxLong =  geoJSON.longitude;
console.log(approxLong);

// configs
var threshold = 0.70;

let labels = ["Tyler Odenthal"];
let ASA_id = '478549868'
let fps;
let lastSeen = 'Waiting';
let lastTime = 'Waiting';
let lastSeenArray = [];
let frameCounter = 0;
const times = [];

var video = document.getElementById('webcam');
var liveView = document.getElementById('liveView');
var webcamSection = document.getElementById('webcamSection');
var enableWebcamButton = document.getElementById('webcamButton');
var ZoomButtons = document.getElementById('ZoomButtons');
var sliderContent = document.getElementById('sliderContent');

var pos = document.getElementById("pos");

function submit() {
    
    // Get elements for Algorand and Camera information
    var AlgoContent = document.getElementById('AlgoContent');
    var AlgoAddress = document.getElementById('AlgoAddress');
    var CameraName = document.getElementById('CameraName');
    var RTSPName = document.getElementById('RTSPName');
    
    // Hide Algorand Input Boxes
    AlgoContent.style.display = "none";
    
    // Create Section for Inputed Algorand Information  
    var NewAlgoSection = document.getElementById('NewAlgoSection');
    var SetAlgoContent = document.createElement("p");
    
    // Append P element to New Algorand Section
    NewAlgoSection.appendChild(SetAlgoContent);

    // Show Webcam Section
    webcamSection.removeAttribute("hidden");

    // Run HTTP Get Request 
    url = 'https://node.algoexplorerapi.io/v2/accounts/' + AlgoAddress.value + '/assets/' + ASA_id;
    response = httpGet(url);
    console.log(response);

    if (responseStatus == 200) {
        var obj = JSON.parse(response)
        var accountBalance = obj["asset-holding"]["amount"];
        console.log(accountBalance);
    } else {
        var accountBalance = 0;
    }
    
    if (accountBalance > 0) {
        var walletConnected = "Successful";
        SetAlgoContent.innerText = 'Wallet Connected: ' + walletConnected + '\n\n' + 'Algorand Address: ' + AlgoAddress.value + '\n\n' + 'Approximate City: ' + approxCity + '\n\n' + 'Camera Name: ' + CameraName.value + '\n\n' + 'BIRDS Balance: ' + accountBalance.toLocaleString("en-US") + '\n\n' + 'RTSP URL: ' + RTSPName.value;
        SetAlgoContent.style.backgroundColor = "rgba(0, 155, 0, 0.6)";
        SetAlgoContent.style.padding = "20px";
    } else {
        var walletConnected = "Not Connected";
        SetAlgoContent.innerText = 'Wallet Connected: ' + walletConnected + '\n\n' + 'Camera Name: ' + CameraName.value + '\n\n' + 'RTSP URL: ' + RTSPName.value;
        SetAlgoContent.style.backgroundColor = "rgba(240, 55, 33, 0.6)";
        SetAlgoContent.style.padding = "20px";
    }
    
}

// Do an HTTP Get Request
function httpGet(theUrl) {
    let xmlHttpReq = new XMLHttpRequest();
    xmlHttpReq.open("GET", theUrl, false); 
    xmlHttpReq.send(null);
    responseStatus = xmlHttpReq.status;
    return xmlHttpReq.responseText;
}

function zoomIn() {
    camWidth = 1080;
    camHeight = 1080;
    video.style.height = camWidth + 'px'
    video.style.width = camHeight + 'px'
}

function zoomOut() {
    camWidth = 720;
    camHeight = 720;
    video.style.height = camWidth + 'px'
    video.style.width = camHeight + 'px'
}

var slider = document.getElementById("vidScaleRange");
var output = document.getElementById("scaleOutput");
output.innerHTML = slider.value;

var camWidth = 720;
var camHeight = 720;
var screenWidth = screen.width;
var screenHeight = screen.height;
var screenPadding = 30;
var screenRatio = slider.value;

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}
    
// Enable the live webcam view and start classification.
function enableCam(event) {
    // Only continue if model loads.
    if (!model) {
        return;
    }
    
    // Hide the button once clicked.
    event.target.classList.add('removed');
    ZoomButtons.removeAttribute("hidden");
    sliderContent.removeAttribute("hidden");

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia({audio: true, video: { width: camWidth, height: camHeight, facingMode: 'environment' }}).then(function(stream) {
        video.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding) + 'px;'
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
    });
}

// Store the resulting model in the global scope of our app.
var model = undefined;

// Wait for it to finish loading. Machine Learning models 
// can be large and take a moment to get everything needed to run.

async function loadModel() {
    model = await tf.loadGraphModel('https://raw.githubusercontent.com/BirdBotASA/BirdBotSoftware/main/best_web_model/model.json');
    console.log("model loaded");
    return model
};

// load Model
model = loadModel();

var children = [];

video.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding) + 'px;'

function predictWebcam() {
    // Now let's start classifying a frame in the stream.

    // Get current time
    const now = performance.now();
    var current = new Date();
    let lastSeenForArray = null;

    current.toLocaleTimeString();

    tf.engine().startScope();

    output.innerHTML = slider.value;

    let [modelWidth, modelHeight] = model.inputs[0].shape.slice(1, 3);

    const input = tf.tidy(() => {
        return tf.image
            .resizeBilinear(tf.browser.fromPixels(video), [modelWidth, modelHeight])
            .div(255.0)
            .expandDims(0);
    });

    input.dtype = "float32";

    predictions = model.executeAsync(input).then(predictions=> {

        const bboxes = predictions[0].dataSync();
        //console.log('Bounding Boxes: ', bboxes);
        
        const scores = predictions[1].dataSync();
        //console.log('Confidence Scores: ', scores);

        const classes = predictions[2].dataSync();
        //console.log('Classes: ', classes);

        // Remove any highlighting we did previous frame.
        for (let i = 0; i < children.length; i++) {
            liveView.removeChild(children[i]);
        }
        
        children.splice(0);

        slider.oninput = function() {
            output.innerHTML = this.value;
            screenRatio = this.value;
            adjustedWidth = ((screenWidth / screenRatio) + screenPadding);
            video.style = 'margin-left: ' + adjustedWidth + 'px;'
            video.style.height = camWidth + 'px'
            video.style.width = camHeight + 'px'
            console.log(adjustedWidth);
        }
        
        // Construct the FPS Visual Widget
        const currentFPS = document.createElement('p');

        currentFPS.innerText = 'FPS: ' + fps;
                
        currentFPS.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding) + 'px;';

        liveView.appendChild(currentFPS);

        children.push(currentFPS);

        // Construct the Last Seen Visual Widget
        const lastSeenWidget = document.createElement('p');

        lastSeenWidget.innerText = 'Last Seen: ' + lastSeen + ' - ' + lastTime;
                
        lastSeenWidget.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding + 70) + 'px;';

        liveView.appendChild(lastSeenWidget);

        children.push(lastSeenWidget);

        // Now lets loop through predictions.
        for (let n = 0; n < scores.length; n++) {

            // If over threshold % classify it and draw
            if (scores[n] > threshold) {
                
                const currentClass = document.createElement('p');

                math1 = (n*4)
                x1 = bboxes[math1]

                math2 = (n*4)+1
                y1 = bboxes[math2]

                math3 = (n*4)+2
                x2 = bboxes[math3]
                
                math4 = (n*4)+3
                y2 = bboxes[math4]

                x1 *= camWidth;
                x2 *= camWidth;
                y1 *= camHeight;
                y2 *= camHeight;
                const boxWidth = x2 - x1;
                const boxHeight = y2 - y1;

                lastSeen = labels[classes[n]];

                lastSeenForArray = labels[classes[n]];

                lastTime = current.toLocaleTimeString();

                currentClass.innerText = labels[classes[n]] + ' - with ' 
                        + Math.round(parseFloat(scores[n]) * 100) 
                        + '% confidence.';
                
                currentClass.style = 'margin-left: ' + (x1 + (screenWidth / screenRatio) + screenPadding) 
                + 'px; margin-top: ' + (y1 - 10) 
                + 'px; width: '	+ (boxWidth - 10) 
                + 'px; top: 0; left: 0;';

                const highlighter = document.createElement('div');
                highlighter.setAttribute('class', 'highlighter');
                highlighter.style = 'left: ' + (x1 + (screenWidth / screenRatio) + screenPadding) 
                + 'px; top: ' + y1 
                + 'px; width: ' + boxWidth
                + 'px; height: ' + boxHeight + 'px;';

                liveView.appendChild(highlighter);
                liveView.appendChild(currentClass);
                children.push(highlighter);
                children.push(currentClass);
            }
        }
        
        tf.engine().endScope();

        while (times.length > 0 && times[0] <= now - 1000) {
            times.shift();
        }

        times.push(now);
        fps = times.length;
        frameCounter += 1;

        if (lastSeenForArray !== null) {
            lastSeenArray.push(lastSeenForArray);
        }


        if (frameCounter > (fps*2)) {
            frameCounter = 0;
            console.log(lastSeenArray);
            lastSeenArray = [];
            console.log('Frame Counter Reset');
        } 
        
        // Call this function again to keep predicting when the browser is ready.
        window.requestAnimationFrame(predictWebcam);
    });
}
