// Import dependencies
import React, { useState} from 'react';

import { Button, StyleSheet, Text, View } from 'react-native';
import Canvas, {Image as CanvasImage}  from 'react-native-canvas';
import {
  Image,
  ImageUtil,
  media,
  MobileModel,
  Module,
  Tensor,
  torch,
  torchvision
} from 'react-native-pytorch-core';
import { Buffer } from "buffer";
// import { useSafeAreaInsets } from 'react-native-safe-area-context';

var similarity = require( 'compute-cosine-similarity' );


const T = torchvision.transforms;

import * as fs from 'expo-file-system';

import * as wav from 'node-wav';


const COMPUTE_FEATURES_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/compute_features.ptl';

const MEAN_VAR_NORM_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/mean_var_norm.ptl';

const EMBEDDING_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/embedding.ptl';

const LANDMARKS_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/facial_landmarks.ptl';

const FACE_IMAGE_URL = 
  'https://raw.githubusercontent.com/tiqq111/mediapipe_pytorch/main/facial_landmarks/4.jpg';


// Variable to hold a reference to the loaded ML model
let cfModel: Module | null = null;
let mvnModel: Module | null = null;
let eModel: Module | null = null;

let flModel: Module | null = null;
let imageTensor: Tensor | null = null;
let prediction1: Tensor | null = null;
let prediction2: Tensor | null = null;
let landmarks: Tensor | null = null;



// App function to render a camera and a text
export default function App() {
  // Safe area insets to compensate for notches and bottom bars
  // const insets = useSafeAreaInsets();
  // Create a React state to store the top class returned from the
  // classifyImage function
  const [similarityState, setSimilarityState] = useState(
    "Similarity value here",
  );

  const [landmarksFound, setLandmarksFound] = useState<boolean>(
    false,
  );


  async function loadModels(){
    if (cfModel === null) {
      const cfFilePath = await MobileModel.download(COMPUTE_FEATURES_MODEL_URL);
      cfModel = await torch.jit._loadForMobile(cfFilePath);
      console.log('Compute features model saved');
    }

    if (mvnModel === null) {
      const mvnFilePath = await MobileModel.download(MEAN_VAR_NORM_MODEL_URL);
      mvnModel = await torch.jit._loadForMobile(mvnFilePath);
      console.log('Mean var norm model saved');
    }

    if (eModel === null) {
      const eFilePath = await MobileModel.download(EMBEDDING_MODEL_URL);
      eModel = await torch.jit._loadForMobile(eFilePath);
      console.log('Embedding model saved');
    }

    if (flModel === null) {
      const flFilePath = await MobileModel.download(LANDMARKS_MODEL_URL);
      flModel = await torch.jit._loadForMobile(flFilePath);
      console.log('Facial landmarks model saved');
    }

    console.log('Load models finished.')
  }



  // Function to handle images whenever the user presses the capture button
  async function handleAudio(url: string) : Promise<Tensor>  {
    console.log('begin audio handling...');


    let wavUri: string;

    await fs.downloadAsync(url, fs.documentDirectory + url.split('/').pop())
    .then(({ uri }) => {
      console.log('Finished downloading to ', uri);
      wavUri = uri;
    })
    .catch(error => {
      console.error(error);
    });


    console.log('promised')

    console.log('wavuri', wavUri);
    var audioData = await fs.readAsStringAsync(wavUri, {"encoding":"base64"})

    // console.log('audio data', audioData);
    var buffer = Buffer.from(audioData, 'base64')

    var result = wav.decode(buffer);
    var channelData: Array<Float32Array> = result.channelData;

    console.log('sample rate', result.sampleRate);
    console.log('channel data', channelData[0].length);
    console.log('max data', channelData[0].reduce((a, b) => Math.max(a, b)));
    console.log('min data', channelData[0].reduce((a, b) => Math.min(a, b)));
    // console.log(result.channelData); // array of Float32Arrays

    console.log('type of', typeof channelData);
    console.log('type of', typeof channelData[0]);
    console.log('type of', typeof channelData[0].values());
    console.log('type of', typeof Array(channelData));
    console.log('type of', typeof Array(channelData[0]));

  

    var audioTensor = torch.tensor(Array.from(channelData[0])).reshape([1, channelData[0].length])

    // console.log('audio tensor', audioTensor.data())
    console.log('audio tensor shape', audioTensor.shape)

    var cfOutput = await cfModel.forward<Tensor, Tensor>(audioTensor);

    console.log('cf output shape', cfOutput.shape)

    var mvnOutput = await mvnModel.forward<Tensor, Tensor>(cfOutput);

    console.log('mvn output shape', mvnOutput.shape)

    var output = await eModel.forward<Tensor, Tensor>(mvnOutput);
    console.log('output shape', output.shape)
    // console.log('output', output.data())

    console.log('Audio prediction finished.')

    return output;

  }

  async function makePrediction1(){
    const url1 = 'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/grabacion3.wav';
    prediction1 = await handleAudio(url1);
  }

  async function makePrediction2(){
    const url2 = 'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/example.wav';
    prediction2 = await handleAudio(url2);
  }

  async function computeSimilarity(){
    var s = similarity(Array.from(prediction1.data()), Array.from(prediction2.data()));
    setSimilarityState('Similarity: '+ s.toFixed(5));
  }

  async function makeImagePrediction(){
    // const randomTensor = torch.randn([1, 3, 192, 192]);

    const image: Image = await ImageUtil.fromURL(FACE_IMAGE_URL);
    const width = image.getWidth();
    const height = image.getHeight();
    const blob = media.toBlob(image);
    let tensor = torch.fromBlob(blob, [height, width, 3]);
    tensor = tensor.permute([2, 0, 1]);
    tensor = tensor.div(127.5).add(-1.0);
    const centerCrop = T.centerCrop(Math.min(width, height));
    tensor = centerCrop(tensor);
    const resize = T.resize(192);
    tensor = resize(tensor);
    imageTensor = tensor;
    tensor = tensor.unsqueeze(0);
    console.log('shape', tensor.shape);

    var flOutput = await flModel.forward<Tensor, Tensor>(tensor);
    console.log('facial landmarks output', flOutput[0].shape);
    // console.log('confidence output', flOutput[1].shape);

    landmarks = flOutput[0].reshape([-1, 3]);

    // console.log('landmarks', landmarks.data())
    console.log('landmarks shape', landmarks.shape)

    setLandmarksFound(true);

  }

  function handleCanvas(canvas: Canvas) {

    if (!(canvas instanceof Canvas)) {
      return;
    }
    const canvasImage = new CanvasImage(canvas);
    canvas.width = 500;
    canvas.height = 500;
    canvasImage.src = FACE_IMAGE_URL;
    // 
      if (landmarksFound && landmarks.data().length>0){
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'purple';
        // var buffer = new Uint8ClampedArray(192 * 192 * 4);
        // for(var y = 0; y < 192; y++) {
        //   for(var x = 0; x < 192; x++) {
        //       var pos = (y * 192 + x) * 4; // position in buffer based on x and y
        //       buffer[pos  ] = imageTensor[0][y][x].data()[0];           // some R value [0, 255]
        //       buffer[pos+1] = imageTensor[1][y][x].data()[0];           // some G value
        //       buffer[pos+2] = imageTensor[2][y][x].data()[0];           // some B value
        //       buffer[pos+3] = 255;           // set alpha channel
        //   }
        // }

        // // create imageData object
        // var idata = ctx.createImageData(192, 192);

        // // set our buffer as source
        // idata.data.set(buffer);

        // // update canvas with new data
        // ctx.putImageData(idata, 0, 0);
        ctx.drawImage(canvasImage, 0, 0, 500, 500);

        for (let i = 0; i < 468; i++) {
          // console.log(landmarks.data()[i*3], landmarks.data()[i*3+1])
          ctx.fillRect(landmarks.data()[i*3]*2, landmarks.data()[i*3+1]*2, 10, 10);


      }  

  }}

  // useEffect(() => {

  //       }) 

  // function componentDidUpdate() {
  //   // Get the canvas object from the ref
  //   const canvas = this.canvas.current;
  //   const ctx = canvas.getContext("2d");
  //   ctx.fillStyle = this.state.color;
  //   ctx.fillRect(0, 0, 100, 100);
  // }

  return (
  // <SafeAreaProvider>
    <View style={StyleSheet.absoluteFill}>
      <Button title='Load models' onPress={loadModels} />
      <Button title='Make prediction 1' onPress={makePrediction1} />
      <Button title='Make prediction 2' onPress={makePrediction2} />
      <Button title='Compute similarity' onPress={computeSimilarity} />
      <Button title='Make Image Prediction' onPress={makeImagePrediction} />
      <View style={styles.labelContainer}>
        {/* Change the text to render the top class label */}
        <Text style={[styles.setFontSizeFour]}>{similarityState}</Text>
      </View>
      <View style={{ flex: 1 }}>
        <Canvas style={{ width: '100%', height: '100%', backgroundColor: 'black' }} ref={handleCanvas} />
      </View>
    </View>
  // </SafeAreaProvider>
  );
}

// Custom render style for label container
const styles = StyleSheet.create({
  labelContainer: {
    padding: 20,
    margin: 20,
    marginTop: 40,
    borderRadius: 10,
    backgroundColor: 'white',
  },
  setFontSizeOne: {
    fontSize: 15 // Define font size here in Pixels
  },
  setFontSizeTwo: {
    fontSize: 20 // Define font size here in Pixels
  },
  setFontSizeThree: {
    fontSize: 25 // Define font size here in Pixels
  },
  setFontSizeFour: {
    fontSize: 30 // Define font size here in Pixels
  },
});
