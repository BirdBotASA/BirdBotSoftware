geoUrlAPI = "https://json.geoiplookup.io/";

geoResponse = httpGet(geoUrlAPI);
const geoJSON = JSON.parse(geoResponse);
// console.log(geoJSON);

const approxCity = geoJSON.city;
// console.log(approxCity);

const approxLat =  geoJSON.latitude;
// console.log(approxLat);

const approxLong =  geoJSON.longitude;
// console.log(approxLong);

// configs
var threshold = 0.70;

let labels = ["Abert's Towhee", "Allen's Hummingbird", "Ruby-throated Hummingbird", "Rufous Hummingbird", "American Crow", "Northern Mockingbird", "American Goldfinch", "Pine Siskin", "House Finch - Female", "Chestnut-backed Chickadee", "House Sparrow", "Golden-crowned Sparrow", "Black-capped Chickadee", "Steller's Jay", "American Robin", "Eastern Towhee", "American Tree Sparrow", "Anna's Hummingbird", "Ash-throated Flycatcher", "Bald Eagle", "Baltimore Oriole", "Barn Swallow", "Cedar Waxwing", "Bewick's Wren", "Verdin", "Black_Phoebe", "House Wren", "Mountain Chickadee", "Dark-eyed Junco", "Red-breasted Nuthatch", "House Finch - Male", "Townsend's Warbler", "European Starling", "Black-chinned Hummingbird", "Black-crested Titmouse", "Black-headed Grosbeak", "Black-headed Grosbeak - Female", "White-crowned Sparrow", "Red-winged Blackbird", "Brown-headed Cowbird", "Western Bluebird", "Common Grackle", "Bullock's Oriole", "Black-billed Magpie", "Black Bear", "Black Vulture", "Common Raven", "Osprey", "Carolina Wren", "California Scrub-Jay", "Turkey Vulture", "Northern Cardinal", "Eastern Phoebe", "Blue Jay", "Brewer's Blackbird", "Great-tailed Grackle", "California Towhee", "Grey Catbird", "Rock Pigeon", "Broad-tailed Hummingbird", "Bushtit", "Eurasian Collared-Dove", "Western Tanager", "White-breasted Nuthatch", "Pine Warbler", "Common Yellowthroat", "Cooper's Hawk", "Cactus Wren", "Say's Phoebe", "Song Sparrow", "Carolina Chickadee", "Ruby-crowned Kinglet", "Red-bellied Woodpecker", "Red-headed Woodpecker", "Chimney Swift", "Chipping Sparrow", "Clay-colored Sparrow", "Common Redpoll", "Curve-billed Thrasher", "Purple Finch", "Northern Flicker", "Downy Woodpecker", "Tufted Titmouse", "Eastern Bluebird", "Eastern Kingbird", "Eastern Meadowlark", "Mourning Dove", "Gambel's Quail", "Gila Woodpecker", "Golden-fronted Woodpecker", "Horned Lark", "House Fire", "Indigo Bunting", "Killdeer", "Ladder-backed Woodpecker", "Lesser Goldfinch", "Loggerhead Shrike", "Mountain Bluebird", "Oak Titmouse", "Orange-crowned Warbler", "Orchard Oriole", "Osprey - Chick", "Osprey - Egg", "Painted Bunting", "Phainopepla", "Red-shouldered Hawk", "Western Kingbird", "Yellow_rumped_Warbler", "Spotted Towhee", "Savannah Sparrow", "Western Meadowlark", "Wild Boar", "Yellow Warbler"];
let ASA_id = '478549868'
let fps;
let fpsTag = document.getElementById('FPSTag');
let lastSeen = 'Waiting';
let lastTime = 'Waiting';
let lastSeenArray = [];
let frameCounter = 0;
const times = [];

var video = document.getElementById('webcam');
var liveView = document.getElementById('liveView');
var cameraButtonSection = document.getElementById('cameraButtonSection');
var enableWebcamButton = document.getElementById('webcamButton');
var ZoomButtons = document.getElementById('ZoomButtons');
var sliderContent = document.getElementById('sliderContent');
var audioOn = true;
var videoOn = true;

var pos = document.getElementById("pos");

function submit() {
    
    // Get elements for Algorand and Camera information
    var AlgoContent = document.getElementById('AlgoContent');
    var AlgoAddress = document.getElementById('AlgoAddress');
    var CameraName = document.getElementById('CameraName');
    var RTSPName = document.getElementById('RTSPName');
    var RTSPButton = document.getElementById('rtspButton');
    
    // Hide Algorand Input Boxes
    AlgoContent.style.display = "none";
    
    // Create Section for Inputed Algorand Information  
    var NewAlgoSection = document.getElementById('NewAlgoSection');
    var SetAlgoContent = document.createElement("p");
    var SetRTSPContent = document.createElement("p");
    
    // Append P element to New Algorand Section
    NewAlgoSection.appendChild(SetAlgoContent);
    NewAlgoSection.appendChild(SetRTSPContent);

    // Show Webcam Section
    cameraButtonSection.removeAttribute("hidden");

    // Run HTTP Get Request 
    AlgoExplorerAPI = 'https://node.algoexplorerapi.io/v2/accounts/' + AlgoAddress.value + '/assets/' + ASA_id;
    AlgoExplorerResponse = httpGet(AlgoExplorerAPI);
    console.log(AlgoExplorerResponse);

    if (responseStatus == 200) {
        var obj = JSON.parse(AlgoExplorerResponse)
        var accountBalance = obj["asset-holding"]["amount"];
        console.log(accountBalance);
    } else {
        var accountBalance = 0;
    }

    if (RTSPName.value.includes("rtsp://")) {
        var RtspConfirmed = "Valid";
        RTSPButton.removeAttribute("hidden");
        console.log("CONFIRMED RTSP URL");
        SetRTSPContent.innerText = 'RTSP URL: ' + RTSPName.value + '\n\n' + 'Valid URL: ' + RtspConfirmed;
        SetRTSPContent.style.backgroundColor = "rgba(0, 155, 0, 0.6)";
        SetRTSPContent.style.padding = "20px";
    } else {
        var RtspConfirmed = "Not Valid";
        RTSPButton.setAttribute("hidden", "hidden");
        SetRTSPContent.innerText = 'RTSP URL: ' + RTSPName.value + '\n\n' + 'Valid URL: ' + RtspConfirmed;
        SetRTSPContent.style.backgroundColor = "rgba(240, 55, 33, 0.6)";
        SetRTSPContent.style.padding = "20px";
    }
    
    if (accountBalance > 0) {
        var walletConnected = "Wallet Connected: Successful";
        SetAlgoContent.innerText = walletConnected + '\n\n' + 'Algorand Address: ' + truncate(AlgoAddress.value, 8) + '\n\n' + 'Approximate City: ' + approxCity + '\n\n' + 'Camera Name: ' + CameraName.value + '\n\n' + 'BIRDS Balance: ' + accountBalance.toLocaleString("en-US");
        SetAlgoContent.style.backgroundColor = "rgba(0, 155, 0, 0.6)";
        SetAlgoContent.style.padding = "20px";
        
    } else {
        var walletConnected = "Wallet Connected: Unsuccessful";
        SetAlgoContent.innerText = walletConnected + '\n\n' + 'Camera Name: ' + CameraName.value ;
        SetAlgoContent.style.backgroundColor = "rgba(240, 55, 33, 0.6)";
        SetAlgoContent.style.padding = "20px";
    }
    
}

function goBack() {

    // Show Webcam Section
    cameraButtonSection.setAttribute("hidden", "hidden");
    NewAlgoSection.removeChild(NewAlgoSection.childNodes[0]);
    NewAlgoSection.removeChild(NewAlgoSection.childNodes[0]);
    AlgoContent.style.display = "block";

    console.log(ZoomButtons.style.display);

    if (ZoomButtons.visibility === "hidden") {
        ZoomButtons.setAttribute("hidden", "hidden");
        console.log("TRUE");
    } else {
        console.log("FALSE");
        ZoomButtons.setAttribute("hidden", "hidden");
        cameraButtonSection.setAttribute("hidden", "hidden");
        sliderContent.setAttribute("hidden", "hidden");
        videoOn = false;
    }

}

function audioSwitch() {
    // Get the checkbox
    var AudioToggle = document.getElementById('AudioToggle');
    var audioOutput = document.getElementById('AudioToggle')

    audioOutput.innerHTML = AudioToggle.checked;

    // If the checkbox is checked, display the output text
    if (AudioToggle.checked == true){
        console.log("Toggle On");
        var audioOn = true;
        video.srcObject.getAudioTracks()[0].enabled = audioOn; // or false to mute it.
    } else {
        console.log("Toggle Off");
        var audioOn = false;
        video.srcObject.getAudioTracks()[0].enabled = audioOn; // or false to mute it.
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

function truncate(str, n) {
    return (str.length > n) ? str.substr(0, n-1) + '...' : str;
};

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

var vidSlider = document.getElementById("vidScaleRange");
var vidSliderOutput = document.getElementById("vidScaleOutput");
vidSliderOutput.innerHTML = vidSlider.value;

var confSlider = document.getElementById("confScaleRange");
var confSliderOutput = document.getElementById("confScaleOutput");
confSliderOutput.innerHTML = confSlider.value;

var camWidth = 720;
var camHeight = 720;
var screenWidth = screen.width;
var screenHeight = screen.height;
var screenPadding = 30;
var screenRatio = vidSlider.value;

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
    cameraButtonSection.setAttribute("hidden", "hidden");
    ZoomButtons.removeAttribute("hidden");
    sliderContent.removeAttribute("hidden");

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia({audio: audioOn, video: { width: camWidth, height: camHeight, facingMode: 'user' }}).then(function(stream) {
        video.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding) + 'px;'
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
    });
}

//////////////////////////////////////////////////////////
//                     WORKING HERE                     //
// Enable the live webcam view and start classification.//
//                                                      //
//////////////////////////////////////////////////////////
function enableRTSP() {
    // Only continue if model loads.
    if (!model) {
        return;
    }
    
    // Hide the button once clicked.
    cameraButtonSection.setAttribute("hidden", "hidden");
    ZoomButtons.removeAttribute("hidden");
    sliderContent.removeAttribute("hidden");

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia({audio: audioOn, video: { width: camWidth, height: camHeight, facingMode: 'user' }}).then(function(stream) {
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
let lastAwardedSpecies = [];
let emptyArrayCounter = 0;
let maxSeen = 0;

video.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding) + 'px;'

function predictWebcam() {
    // Now let's start classifying a frame in the stream.
    if (videoOn === false) {
        
        return
    }

    // Get current time
    const now = performance.now();
    var current = new Date();
    let lastSeenForArray = null;

    current.toLocaleTimeString();

    vidSliderOutput.innerHTML = vidSlider.value;
    confSliderOutput.innerHTML = confSlider.value;

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

        if (maxSeen < scores.length) {
            maxSeen = scores.length;
            console.log(maxSeen);
        }

        // Remove any highlighting we did previous frame.
        for (let i = 0; i < children.length; i++) {
            liveView.removeChild(children[i]);
        }
        
        children.splice(0);

        confSlider.oninput = function() {
            confSliderOutput.innerHTML = this.value;
            threshold = (this.value / 100);
            console.log(threshold);
        }
        
        vidSlider.oninput = function() {
                vidSliderOutput.innerHTML = this.value;
                screenRatio = this.value;
                adjustedWidth = ((screenWidth / screenRatio) + screenPadding);
                video.style = 'margin-left: ' + adjustedWidth + 'px;'
                video.style.height = camWidth + 'px'
                video.style.width = camHeight + 'px'
                console.log(adjustedWidth);
        }

        // Construct the FPS Visual Widget
        fpsTag.innerText = 'FPS: ' + fps;
        
        fpsTag.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding + 20) + 'px;'

        liveView.prepend(fpsTag);

        children.push(fpsTag);

        // Construct the Last Seen Visual Widget
        const lastSeenWidget = document.createElement('p');

        lastSeenWidget.innerText = 'Last Seen: ' + lastSeen + ' - ' + lastTime;
                
        lastSeenWidget.style = 'margin-left: ' + ((screenWidth / screenRatio) + screenPadding + 90) + 'px;'

        liveView.prepend(lastSeenWidget);

        children.push(lastSeenWidget);

        // Now lets loop through predictions.
        for (let n = 0; n < scores.length; n++) {

            // If over threshold % classify it and draw
            if (scores[n] > threshold) {
                
                const currentClass = document.createElement('p');

                math1 = (n*4);
                x1 = bboxes[math1];

                math2 = (n*4)+1;
                y1 = bboxes[math2];

                math3 = (n*4)+2;
                x2 = bboxes[math3];
                
                math4 = (n*4)+3;
                y2 = bboxes[math4];

                x1 *= camWidth;
                x2 *= camWidth;
                y1 *= camHeight;
                y2 *= camHeight;
                const boxWidth = x2 - x1;
                const boxHeight = y2 - y1;

                lastSeen = labels[classes[n]];

                lastSeenForArray = labels[classes[n]];

                lastTime = current.toLocaleTimeString();

                var numOfLastSeen = lastSeenArray.filter(x => x === lastSeen).length;

                // console.log(numOfLastSeen);

                if (numOfLastSeen > fps && !lastAwardedSpecies.includes(lastSeen) || lastAwardedSpecies.length < maxSeen.length) {
                    lastAwardedSpecies.push(lastSeen);
                    console.log("SEND CONFIRMED SPECIES AND EARN BIRDS: " + lastAwardedSpecies);
                }

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
            
            if (lastSeenArray.length === 0) {
                emptyArrayCounter += 1;
                console.log('No Species Seen: ' + emptyArrayCounter);
            }
            
            if (emptyArrayCounter === 3) {
                emptyArrayCounter = 0;
                lastAwardedSpecies = [];
                maxSeen = 0;
                console.log('Species Awarded Reset');
            }

            frameCounter = 0;
            console.log(lastSeenArray);
            lastSeenArray = [];
            console.log('Frame Counter Reset');
        } 
        
        // Call this function again to keep predicting when the browser is ready.
        window.requestAnimationFrame(predictWebcam);
    });
}