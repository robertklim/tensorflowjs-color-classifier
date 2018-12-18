let data;

function preload() {
    data = loadJSON('../data/colorData.json');
}

function setup() {
    createCanvas(400, 400);
    // console.log(data.entries.length);

    let colors = [];

    for (let record of data.entries) {
        let col = [record.r / 255, record.g / 255, record.b / 255]; // normalize color values
        colors.push(col);
    }
    // console.log(colors);

    // load input data as a 2d tensor
    let xs = tf.tensor2d(colors);
    // console.log(xs.shape);

}

function draw() {
    background(0);
}