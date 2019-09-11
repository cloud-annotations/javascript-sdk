# Object Detection JavaScript SDK
The Cloud Annotations Object Detection SDK makes it easy to use your custom trained object detection models in the browser or on the backend with TensorFlow.js. Simply point the object detector to your `model_web` folder and pass the loaded model a `<canvas>`, `<img>` or `<video>` reference.

## Installation
```bash
npm install @cloud-annotations/object-detection
```

## Usage
```js
import objectDetector from '@cloud-annotations/object-detection'

const img = document.getElementById('img')

const model = await objectDetector.load('/model_web')
const predictions = await model.detect(img)

// predictions =>
[{
  bbox: [x, y, width, height],
  class: 'dog',
  score: 0.92
},
{
  bbox: [x, y, width, height],
  class: 'cat',
  score: 0.72
}]
```

Example usage: [Real-Time Object Detection With React](https://github.com/cloud-annotations/object-detection-react).
