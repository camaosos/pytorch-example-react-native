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


const T = torchvision.transforms;


const LANDMARKS_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/facial_landmarks.ptl';

const FACE_IMAGE_URL = 
  'https://raw.githubusercontent.com/tiqq111/mediapipe_pytorch/main/facial_landmarks/4.jpg';


let flModel: Module | null = null;
let imageTensor: Tensor | null = null;
let landmarks: Tensor | null = null;



// App function to render a camera and a text
export default function App() {
  // Safe area insets to compensate for notches and bottom bars
  // const insets = useSafeAreaInsets();
  // Create a React state to store the top class returned from the
  // classifyImage function


  const [landmarksFound, setLandmarksFound] = useState<boolean>(
    false,
  );

  async function loadModels(){


    if (flModel === null) {
      const flFilePath = await MobileModel.download(LANDMARKS_MODEL_URL);
      flModel = await torch.jit._loadForMobile(flFilePath);
      console.log('Facial landmarks model saved');
    }

    console.log('Load models finished.')
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
        ctx.drawImage(canvasImage, 0, 0, 500, 500);

        for (let i = 0; i < 468; i++) {
          ctx.fillRect(landmarks.data()[i*3]*2, landmarks.data()[i*3+1]*2, 10, 10);


      }  

  }}


  return (
    <View>
      <Text style={[{fontSize: 30 }]}>Face</Text>  
      <Button title='Load models' onPress={loadModels} />

      <Button title='Make Image Prediction' onPress={makeImagePrediction} />

      <View>
        <Canvas style={{ width: '100%', height: '100%', backgroundColor: 'black' }} ref={handleCanvas} />
      </View>
    </View>
  );
}

