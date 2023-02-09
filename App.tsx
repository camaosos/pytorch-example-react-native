// Import dependencies
import React from 'react';
import { StyleSheet, View} from 'react-native';

import AudioComponent from './components/AudioComponent';
import FaceComponent from './components/FaceComponent';


// App function to render a camera and a text
export default function App() {
  // Safe area insets to compensate for notches and bottom bars
  // const insets = useSafeAreaInsets();
  // Create a React state to store the top class returned from the
  // classifyImage function

  return (
  // <SafeAreaProvider>
    <View style={StyleSheet.absoluteFill}>

      {/* <AudioComponent/> */}
      <FaceComponent/>

    </View>
  // </SafeAreaProvider>
  );
}
