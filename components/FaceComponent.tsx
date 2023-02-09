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
import * as FaceDetector from 'expo-face-detector';
import ImageEditor from "@react-native-community/image-editor";
import * as fs from 'expo-file-system';

var similarity = require( 'compute-cosine-similarity');

type ImageOffset = {
  x: number,
  y: number,
};

type ImageSize = {
  width: number,
  height: number,
};

type ImageCropData = {
  offset: ImageOffset,
  size: ImageSize,
  displaySize: ImageSize,
  resizeMode: any,
};

type PredictionResult ={
  "landmarks": Tensor,
  "embedding": Tensor
};


const T = torchvision.transforms;


const LANDMARKS_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/facial_landmarks.ptl';

let FACE_IMAGE_URL1 =  // Old Tom Cruise
  'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/tom-cruise-fotogramas-1653393455.jpg';

let FACE_IMAGE_URL2 =  // Miles Fisher
'https://www.themoviedb.org/t/p/w500/shf8LrQ1hmXut0PoYjNHjFuOwqk.jpg';

let FACE_IMAGE_URL3 =  // Football fan
  'https://raw.githubusercontent.com/tiqq111/mediapipe_pytorch/main/facial_landmarks/4.jpg';

let FACE_IMAGE_URL4 =  // Young Tom Cruise
  'https://i.pinimg.com/564x/b4/3c/19/b43c198da72456a873b7ceda2c81f14f.jpg';

let FACE_IMAGE_URL5 =  // Will Smith
  'https://m.media-amazon.com/images/M/MV5BNTczMzk1MjU1MV5BMl5BanBnXkFtZTcwNDk2MzAyMg@@._V1_FMjpg_UX1000_.jpg'

let FACE_IMAGE_URL6 =  // Eugenio Derbez
  'https://images.ctfassets.net/86mn0qn5b7d0/7LLTP9mkGN6VPFqvS6Jiut/030130381554bfe98ce90d125eaa883b/9658-Dora036-EugenioDerbez.jpg'

let FACE_IMAGE_URL7 =  // Serious Tom Cruise
  'https://imagez.tmz.com/image/8e/4by3/2020/12/17/8e450ecb28d54e85865806ac274c2982_md.jpg'

let FACE_IMAGE_URL8 =  // Young me
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/young_me.jpg'

let FACE_IMAGE_URL9 =  // Recent me
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/recent_me.jpg'

const MTCNN_MODEL_URL =
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/mtcnn.ptl';

const RESNET_MODEL_URL =
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/resnet.ptl';

let flModel: Module | null = null;
let landmarks1: Tensor | null = null;
let landmarks2: Tensor | null = null;

let mtcnnModel: Module | null = null;
let resnetModel: Module | null = null;

let embedding1: Tensor | null = null;
let embedding2: Tensor | null = null;


// App function to render a camera and a text
export default function App() {
  // Safe area insets to compensate for notches and bottom bars
  // const insets = useSafeAreaInsets();

  const [landmarksFound, setLandmarksFound] = useState<boolean>(
    false,
  );

  const [similarityState, setSimilarityState] = useState(
    "Similarity value here",
  );

  const [canvasImageData, setCanvasImageData] = useState(
    'data:image/jpeg;base64,',
  );

  async function loadModels(){

    if (flModel === null) {
      const flFilePath = await MobileModel.download(LANDMARKS_MODEL_URL);
      flModel = await torch.jit._loadForMobile(flFilePath);
      console.log('Facial landmarks model saved');
    }

    if (mtcnnModel === null) {
      const mtcnnFilePath = await MobileModel.download(MTCNN_MODEL_URL);
      mtcnnModel = await torch.jit._loadForMobile(mtcnnFilePath);
      console.log('MTCNN model saved');
    }
    if (resnetModel === null) {
      const resnetFilePath = await MobileModel.download(RESNET_MODEL_URL);
      resnetModel = await torch.jit._loadForMobile(resnetFilePath);
      console.log('Resnet model saved');
    }

    console.log('Load models finished.')
  }

  async function findImageLandmarks(uri: string, type:string='url') : Promise<PredictionResult>{
    // const randomTensor = torch.randn([1, 3, 192, 192]);

    var image: Image;
    if (type=='url'){
      image = await ImageUtil.fromURL(uri);
    }
    else if (type=='file'){
      image = await ImageUtil.fromFile(uri);
    }
    else {
      console.log('Unsupported type');
    }

    
    const width = image.getWidth();
    const height = image.getHeight();
    const blob = media.toBlob(image);
    let tensor = torch.fromBlob(blob, [height, width, 3]);
    tensor = tensor.permute([2, 0, 1]);
    tensor = tensor.add(-127.5).div(128);
    const centerCrop = T.centerCrop(Math.min(width, height));
    tensor = centerCrop(tensor);
    
    const flResize = T.resize(192);
    let flTensor = flResize(tensor);
    console.log('image tensor 192', flTensor.data())

    const rnResize = T.resize(160);
    let rnTensor =  rnResize(tensor);
    console.log('image tensor 160', rnTensor.data())




    // imageTensor = tensor;
    // tensor = tensor.unsqueeze(0);
    // console.log('shape', tensor.shape);

    var flOutput = await flModel.forward<Tensor, Tensor>(flTensor.unsqueeze(0));
    console.log('facial landmarks output', flOutput[0].shape);

    let rnOutput = await resnetModel.forward<Tensor, Tensor>(rnTensor.unsqueeze(0));
    console.log('Resnet Output', rnOutput.shape)
    console.log(rnOutput.data())

    return {"landmarks": flOutput[0].reshape([-1, 3]), 
            "embedding": rnOutput};

  }

  async function makeImageCrop(url: string): Promise<string>{

    let fileUri = (await fs.downloadAsync(url, fs.documentDirectory + url.split('/').pop())).uri
    let detectionResult = await FaceDetector.detectFacesAsync(fileUri)
    let boundingBox = detectionResult.faces[0].bounds;

    let cropData: ImageCropData = {
      offset: {x: boundingBox.origin.x, y: boundingBox.origin.y},
      size: {width: boundingBox.size.width, height: boundingBox.size.height},
      displaySize: {width: boundingBox.size.width, height: boundingBox.size.height},
      resizeMode: 'contain/cover/stretch',
    };

    console.log('crop data', cropData)

    var croppedImageUri:string = await ImageEditor.cropImage(fileUri, cropData);
    var data = await fs.readAsStringAsync(croppedImageUri, {"encoding":"base64"})

    setCanvasImageData('data:image/jpeg;base64,'+data);

    return croppedImageUri.substring(7); 
  }


  async function getImageEmbedding1(){
    var imageUri1 = await makeImageCrop(FACE_IMAGE_URL4);
    var result1 = await findImageLandmarks(imageUri1, 'file')
    landmarks1 = result1.landmarks;
    embedding1 = result1.embedding;
  }

  async function getImageEmbedding2(){
    var imageUri2 = await makeImageCrop(FACE_IMAGE_URL8);
    var result2 = await findImageLandmarks(imageUri2, 'file')
    landmarks2 = result2.landmarks;
    embedding2 = result2.embedding;

    setLandmarksFound(true);
  }

  async function computeSimilarity(){
    var s = similarity(Array.from(embedding1.data()), Array.from(embedding2.data()));
    setSimilarityState('Similarity: '+ s.toFixed(5));
    console.log('Similarity');
  }

  async function handleCanvas(canvas: Canvas) {

    if (!(canvas instanceof Canvas)) {
      return;
    }
    const canvasImage = new CanvasImage(canvas);
    canvas.width = 192*2;
    canvas.height = 192*2;
    canvasImage.src = canvasImageData;
    // 
    if (landmarksFound && landmarks2.data().length>0){
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'purple';
      ctx.drawImage(canvasImage, 0, 0, 192*2, 192*2);

      for (let i = 0; i < 468; i++) {
        ctx.fillRect(landmarks2.data()[i*3]*2, landmarks2.data()[i*3+1]*2, 10, 10);


      }   
    }
  }


  return (
    <View>
      <Text style={[{fontSize: 30 }]}>Face</Text>  
      <Button title='Load models' onPress={loadModels} />

      <Button title='Make Image Prediction 1' onPress={getImageEmbedding1} />
      <Button title='Make Image Prediction 2' onPress={getImageEmbedding2} />
      {/* <Button title='Make Image Embedding' onPress={makeImageEmbedding} /> */}
      <Button title='Compute similarity' onPress={computeSimilarity} />
      <View style={styles.labelContainer}>
                {/* Change the text to render the top class label */}
                <Text style={[{fontSize: 30 }]}>{similarityState}</Text>
            </View>

      <View>
        <Canvas style={{ width: '100%', height: '100%', backgroundColor: 'black' }} ref={handleCanvas} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  labelContainer: {
    padding: 20,
    margin: 20,
    marginTop: 40,
    borderRadius: 10,
    backgroundColor: 'white',
  }  });


