import * as tf from '@tensorflow/tfjs'

const calculateMaxScores = (scores, numBoxes, numClasses) => {
  const maxes = []
  const classes = []
  for (let i = 0; i < numBoxes; i++) {
    let max = Number.MIN_VALUE
    let index = -1
    for (let j = 0; j < numClasses; j++) {
      if (scores[i * numClasses + j] > max) {
        max = scores[i * numClasses + j]
        index = j
      }
    }
    maxes[i] = max
    classes[i] = index
  }
  return [maxes, classes]
}

const buildDetectedObjects = (
  width,
  height,
  boxes,
  scores,
  indexes,
  classes,
  labels
) => {
  const count = indexes.length
  const objects = []
  for (let i = 0; i < count; i++) {
    const bbox = []
    for (let j = 0; j < 4; j++) {
      bbox[j] = boxes[indexes[i] * 4 + j]
    }
    const minY = bbox[0] * height
    const minX = bbox[1] * width
    const maxY = bbox[2] * height
    const maxX = bbox[3] * width
    bbox[0] = minX
    bbox[1] = minY
    bbox[2] = maxX - minX
    bbox[3] = maxY - minY
    objects.push({
      bbox: bbox,
      class: labels[parseInt(classes[indexes[i]])],
      score: scores[indexes[i]]
    })
  }
  return objects
}

const runPrediction = async (graph, labels, input) => {
  const batched = tf.tidy(() => {
    const img = tf.browser.fromPixels(input)
    // Reshape to a single-element batch so we can pass it to executeAsync.
    return img.expandDims(0)
  })

  const height = batched.shape[1]
  const width = batched.shape[2]

  const result = await graph.executeAsync(batched)

  const scores = result[0].dataSync()
  const boxes = result[1].dataSync()

  // clean the webgl tensors
  batched.dispose()
  tf.dispose(result)

  const [maxScores, classes] = calculateMaxScores(
    scores,
    result[0].shape[1],
    result[0].shape[2]
  )

  const prevBackend = tf.getBackend()
  // run post process in cpu
  tf.setBackend('cpu')
  const indexTensor = tf.tidy(() => {
    const boxes2 = tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3]])
    return tf.image.nonMaxSuppression(
      boxes2,
      maxScores,
      20, // maxNumBoxes
      0.5, // iou_threshold
      0.5 // score_threshold
    )
  })
  const indexes = indexTensor.dataSync()
  indexTensor.dispose()
  // restore previous backend
  tf.setBackend(prevBackend)

  return buildDetectedObjects(
    width,
    height,
    boxes,
    maxScores,
    indexes,
    classes,
    labels
  )
}

export default {
  load: async path => {
    const graphPath = path + '/model.json'
    const labelsPath = path + '/labels.json'
    const graphPromise = tf.loadGraphModel(graphPath)
    const labelsPromise = fetch(labelsPath).then(data => data.json())
    const [graph, labels] = await Promise.all([graphPromise, labelsPromise])

    return {
      detect: async input => await runPrediction(graph, labels, input)
    }
  }
}
