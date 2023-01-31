import React, { useState} from 'react';
import { Button, StyleSheet, View, Text } from 'react-native';
import {
    MobileModel,
    Module,
    Tensor,
    torch
  } from 'react-native-pytorch-core';
import * as fs from 'expo-file-system';
import * as wav from 'node-wav';
var similarity = require( 'compute-cosine-similarity' );
import { Buffer } from "buffer";


const COMPUTE_FEATURES_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/compute_features.ptl';

const MEAN_VAR_NORM_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/mean_var_norm.ptl';

const EMBEDDING_MODEL_URL = 
  'https://github.com/camaosos/speaker-embedding-pytorch-models/raw/main/embedding.ptl';


// Variable to hold a reference to the loaded ML model
let cfModel: Module | null = null;
let mvnModel: Module | null = null;
let eModel: Module | null = null;
let prediction1: Tensor | null = null;
let prediction2: Tensor | null = null;

export default function AudioComponent() {

    const [similarityState, setSimilarityState] = useState(
        "Similarity value here",
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
    }


      // Function to handle audio whenever the user presses the prediction button
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
        // const channelData: Array<Float32Array> = result.channelData;
        var channelData = result.channelData;

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

    return(

        <View>
            <Text style={[{fontSize: 30 }]}>Audio</Text>
            <Button title='Load audio models' onPress={loadModels} />
            <Button title='Make prediction 1' onPress={makePrediction1} />
            <Button title='Make prediction 2' onPress={makePrediction2} />
            <Button title='Compute similarity' onPress={computeSimilarity} />
            <View style={styles.labelContainer}>
                {/* Change the text to render the top class label */}
                <Text style={[{fontSize: 30 }]}>{similarityState}</Text>
            </View>
        </View>
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