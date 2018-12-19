let data;
let xs, ys;
let model;

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
        inputDim: [3]
    });

    let output = tf.layers.dense({
        units: 9,
        activation: 'softmax'
    });

    model.add(hidden);
    model.add(output);

}

function draw() {
    background(0);
}