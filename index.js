let mobilenet;
let model;
const webcam=new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0;

async function loadMobileNet()
{
    const mobilenet= await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer= mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.input,outputs:layer.output});
}

async function init()
{
    await webcam.setup();
    mobilenet=await loadMobileNet();
    tf.tidy(()=>mobilenet.predict(webcam.capture()))
}

async function train()
{
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 128, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys,
            {
      epochs: 100,
      callbacks:
      {
          onBatchEnd: async (batch, logs) =>
          {
              loss = logs.loss.toFixed(5);
              console.log('LOSS: ' + loss);
          }
      }
  });
}

function doTraining(){
	train();
    alert("Training is done, try out by clicking on the start predicting button to let the browser decide your gestures!");
}

function handleButton(elem)
{
	switch(elem.id)
    {
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);
}

init();