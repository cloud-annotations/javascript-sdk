# Cloud Annotations JavaScript SDK
[![NPM Version](https://img.shields.io/npm/v/@cloud-annotations/models.svg)](https://npmjs.org/package/@cloud-annotations/models)
[![NPM Downloads](https://img.shields.io/npm/dm/@cloud-annotations/models.svg)](https://npmjs.org/package/@cloud-annotations/models)

The Cloud Annotations SDK makes it easy to use your custom trained object detection or classification models in the browser or on the backend with TensorFlow.js. Simply `load()` your `model_web` folder and pass the loaded model a `<canvas>`, `<img>` or `<video>` reference.

## Installation
```bash
npm install @cloud-annotations/models
```

## Usage

### Load a model
```js
import models from '@cloud-annotations/models'

const model = await models.load('/model_web')
```

### Object detection
```js
const img = document.getElementById('img')
const predictions = await model.detect(img)

// predictions =>
[{
  label: 'dog',
  bbox: [x, y, width, height],
  score: 0.92
},
{
  label: 'cat',
  bbox: [x, y, width, height],
  score: 0.72
}]
```

### Classification
```js
const img = document.getElementById('img')
const predictions = await model.classify(img)

// predictions =>
[
  { label: 'dog', score: 0.92 },
  { label: 'cat', score: 0.72 }
]
```

## Usage via Script Tag
No npm install required. Just import via the script tag.
```html
<script src="https://cdn.jsdelivr.net/npm/@cloud-annotations/models"></script>
<script>
  const img = document.getElementById('img')
  models.load('/model_web')
    .then(model => model.detect(img))
    .then(predictions => {
      console.log(predictions)
    })
</script>
```

Example usage: [Real-Time Object Detection With React](https://github.com/cloud-annotations/object-detection-react).
