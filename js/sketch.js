let data;
let xs, ys;
let model;

let rSlider, gSlider, bSlider;

let labelP;

let labelList = [
    'red-ish',
    'green-ish',
    'blue-ish',
    'orange-ish',
    'yellow-ish',
    'pink-ish',
    'purple-ish',
    'brown-ish',
    'grey-ish'
];

function preload() {
    data = loadJSON('../data/colorData.json');
}

function setup() {
    createCanvas(400, 400);
    // console.log(data.entries.length);

    labelP = createP('');

    rSlider = createSlider(0, 255, 255);
    gSlider = createSlider(0, 255, 255);
    bSlider = createSlider(0, 255, 0);

    let colors = [];
    let labels = [];

    for (let record of data.entries) {
        let col = [record.r / 255, record.g / 255, record.b / 255]; // normalize color values
        colors.push(col);
        labels.push(labelList.indexOf(record.label));
    }
    // console.log(colors);
    // console.log(labels);

    // load input data as a 2d tensor
    xs = tf.tensor2d(colors);
    // console.log(xs.shape);

    // load target outputs data as a 2d tensor
    let labelsTensor = tf.tensor1d(labels, 'int32');
    // labelsTensor.print();

    // create one-hot tensor
    ys = tf.oneHot(labelsTensor, 9);
    labelsTensor.dispose();

    // xs.print();
    // ys.print();

    // build the model
    model = tf.sequential();

    let hidden = tf.layers.dense({
        units: 16,
        activation: 'sigmoid',
        inputDim: 3
    });

    let output = tf.layers.dense({
        units: 9,
        activation: 'softmax'
    });

    model.add(hidden);
    model.add(output);

    // create an optimizer
    const learningRate = 0.1;
    const optimizer = tf.train.sgd(learningRate);

    // define optimizer and loss function and compile the model
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy'
    });

    // training
    train().then(results => {
        console.log(results.history.loss);
    });

}

async function train() {
    const trainConfig = {
        epochs: 10,
        validationSplit: 0.1,
        shuffle: true,
        callbacks: {
            onTrainBegin: () => console.log('train begin'),
            onTrainEnd: () => console.log('train end'),
            onBatchEnd: tf.nextFrame,
            onEpochEnd: (num, logs) => {
                console.log(`Epoch: ${num}`);
                console.log(`Loss: ${logs.loss}`);
                console.log(`Validation loss: ${logs.val_loss}`);
            } 
        }
    }
    return await model.fit(xs, ys, trainConfig);
}

function draw() {
    let r = rSlider.value();
    let g = gSlider.value();
    let b = bSlider.value();
    background(r, g, b);

    const xs = tf.tensor2d([
        [r / 255, g / 255, b / 255] // normalize rgb values
    ]);

    let results = model.predict(xs);
    let index = results.argMax(1).dataSync()[0];
    let label = labelList[index];
    labelP.html(label);
    
    
    // stroke(255);
    // strokeWeight(4);
    // line(frameCount % width, 0, frameCount % width, height);
}