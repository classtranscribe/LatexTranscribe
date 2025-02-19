<script setup>
import { ref, onMounted, computed} from "vue";
import axios from 'axios';
import 'katex/dist/katex.min.css'
import katex from 'katex'

const sampleImage = 'src/images/sample.png';
const sampleImageResult = 'src/images/udlmock1result.png';

const showDemoImages = ref(false);
const uploadedFile = ref(null);
const uploadedFileName = ref('');
const showResponse = ref('');
const fileInput = ref(null);
const dragOver = ref(false);

const grayscaleImageUrl = ref('');

const example_json = ref([
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "129.5",
            "80.41",
            "848.55",
            "255.08"
        ],
        "text": "\\begin{array} { l } { { f ( x , y ) = \\displaystyle \\sum _ { n = 0 } ^ { \\infty } \\sum _ { n ^ { \\prime } = 0 } ^ { n } \\frac 1 { n ^ { \\prime } ! ( n - n ^ { \\prime } ) ! } \\, \\frac { \\partial ^ { n } f } { \\partial x ^ { n ^ { \\prime } } \\partial y ^ { n - n ^ { \\prime } } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { n ^ { \\prime } } ( y - y ^ { * } ) ^ { n - n ^ { \\prime } } } } \\\\ { { = \\underbrace { f ( x ^ { * } , y ^ { * } ) } _ { { n = n ^ { \\prime } = 0 } } + \\underbrace { \\frac 1 { 0 ! 1 ! } \\, \\frac { \\partial f } { \\partial x ^ { 0 } \\partial y ^ { \\mathbf { 1 } } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { 0 } ( y - y ^ { * } ) ^ { \\mathbf { 1 } } } _ { { n = \\mathbf { 1 } , \\; n ^ { \\prime } = 0 ; \\; \\frac { \\partial f } { \\partial y } ( x ^ { * } , y ^ { * } ) ( y - y ^ { * } ) }}}} \\end{array}"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "259.97",
            "557.93",
            "741.89",
            "653.72"
        ],
        "text": "\\begin{array} { c } { { + \\underbrace { \\frac { 1 } { 0 ! 2 ! } \\frac { \\partial ^ { 2 } f } { \\partial x ^ { 2 } \\partial y ^ { 0 } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { 2 } ( y - y ^ { * } ) ^ { 0 } } _ { n = 2 , \\ n ^ { \\prime } = 2 ; \\ \\frac { 1 } { 2 } \\frac { \\partial ^ { 2 } f } { \\partial x ^ { 2 } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { 2 } } } } \\end{array} + \\cdot \\cdot \\cdot"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "308.18",
            "458.05",
            "739.47",
            "498.94"
        ],
        "text": "\\begin{array} { r l } & { + \\frac { 1 } { 1 ! 1 ! } \\frac { \\partial ^ { 2 } f } { \\partial x ^ { 1 } \\partial y ^ { 1 } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { 1 } ( y - y ^ { * } ) ^ { 1 } } \\end{array}"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "133.49",
            "248.23",
            "744.0",
            "307.39"
        ],
        "text": "\\begin{array} { r } { + \\frac { 1 } { 1 ! 0 ! } \\frac { \\partial f } { \\partial x ^ { \\mathbf { 1 } } \\partial y ^ { \\mathbf { 0 } } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { \\mathbf { 1 } } ( y - y ^ { * } ) ^ { \\mathbf { 0 } } } \\end{array}"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "260.27",
            "353.89",
            "704.64",
            "394.96"
        ],
        "text": "\\begin{array} { r l } & { + \\frac { 1 } { 2 ! 0 ! } \\frac { \\partial ^ { 2 } f } { \\partial x ^ { 0 } \\partial y ^ { 2 } } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ^ { 0 } ( y - y ^ { * } ) ^ { 2 } } \\end{array}"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "392.45",
            "316.54",
            "675.24",
            "344.87"
        ],
        "text": "n = { \\bf 1 } , \\; n ^ { \\prime } = { \\bf 1 } ; \\; \\frac { \\partial f } { \\partial x } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } )"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "317.74",
            "411.87",
            "643.22",
            "448.48"
        ],
        "text": "\\begin{array} { r } { n = 2 , \\; n ^ { \\prime } = 0 ; \\; { \\frac { 1 } { 2 } } \\, { \\frac { \\partial ^ { 2 } f } { \\partial y ^ { 2 } } } ( x ^ { * } , y ^ { * } ) ( y - y ^ { * } ) ^ { 2 } } \\end{array}"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "351.45",
            "516.57",
            "719.61",
            "549.74"
        ],
        "text": "n = 2 , \\; n ^ { \\prime } = { \\bf 1 } ; \\; \\frac { \\partial ^ { \\, 2 } t } { \\partial x \\partial y } ( x ^ { * } , y ^ { * } ) ( x - x ^ { * } ) ( y - y ^ { * } )"
    },
    {
        "task": "formula_recognition",
        "cls": "formula",
        "bbox": [
            "540.89",
            "225.65",
            "702.52",
            "253.6"
        ],
        "text": "\\frac { \\partial f } { \\partial y } ( x ^ { * } , y ^ { * } ) ( y - y ^ { * } )"
    },
    {
        "task": "base_recognition",
        "cls": "text",
        "bbox": [
            "67.0",
            "33.0",
            "230.0",
            "64.0"
        ],
        "text": "More generally;"
    },
    {
        "task": "base_recognition",
        "cls": "icon",
        "bbox": [
            "390.87",
            "314.08",
            "676.01",
            "344.45"
        ],
        "text": "None"
    },
    {
        "task": "base_recognition",
        "cls": "icon",
        "bbox": [
            "314.61",
            "409.92",
            "648.09",
            "448.89"
        ],
        "text": "None"
    },
    {
        "task": "base_recognition",
        "cls": "icon",
        "bbox": [
            "257.78",
            "353.35",
            "688.86",
            "398.36"
        ],
        "text": "None"
    }
]);

const formula = computed(() => {
  return example_json.value.map(element => {
    const f = element.text;
    return katex.renderToString(f, { throwOnError: false });
  });
})

function toggleResponse(msg) {
  showResponse.value = msg;
}

function selectFile() {
  fileInput.value.click();
}

function onFileChange(event) {
  const image = event.target.files[0];
  if (image) {
    uploadedFile.value = image;
    uploadedFileName.value = image.name;
    console.log(uploadedFile, 'uploaded!');
  }
}

function onDrop(event) {
  event.preventDefault();
  dragOver.value = false;
  const image = event.dataTransfer.files[0];
  if (image) {
    uploadedFile.value = image;
    uploadedFileName.value = image.name;
    console.log(uploadedFile, 'uploaded via drag and drop!');
  }
}

function onDragOver(event) {
  event.preventDefault();
  dragOver.value = true;
}

function onDragLeave() {
  dragOver.value = false;
}

async function submitFile() {
  if (!uploadedFile.value) {
    toggleResponse('No file selected');
    return;
  }
  const formData = new FormData();
  formData.append('file', uploadedFile.value);
  try {
    const response = await axios.post("localhost:8000/upload/", formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Accept': 'image/png', 
      },
      responseType: 'arraybuffer',
    });
    console.log(response.data.message);
    toggleResponse(response.data.message);
  } catch (error) {
    console.error(error);
    toggleResponse("Couldn't upload image");
  }
}

// async function backend_setup() {
//   try {
//     const response = await axios.get('http://127.0.0.1:5000/sanitycheck');
//     toggleResponse(response.data.message);
//   } catch (error) {
//     console.error(error);
//     toggleResponse('Not setup');
//   }
// }

function toggleImages() {
  showDemoImages.value = !showDemoImages.value;
  backend_setup();
}

function deleteFile() {
  if (uploadedFile) {
    uploadedFile.value = null;
    uploadedFileName.value = '';
    fileInput.value.value = '';
    console.log(uploadedFile, 'deleted!');
  }
}
</script>

<template>
  <header style="background-color: rgb(70, 122, 253); color: white; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center;">
    <h1>Model Ensemble Tool</h1>
    <p>To use the tool, please upload an image of the content that you want to be analyzed</p>
    <p>{{ showResponse }}</p>
    <ol>
          <li v-for="(item,ind) in example_json">
              <ul>
                <li>{{ item.task }}</li>
                <li>{{ item.bbox[0] }}, {{ item.bbox[1] }}, {{ item.bbox[2] }}, {{ item.bbox[3] }}</li>
                <li v-html="formula[ind]"></li>
              </ul>
          </li>
      </ol>
  </header>
  <div class="demo">
    <button @click="toggleImages">{{ showDemoImages ? 'Close Demo' : 'Open Demo' }}</button>
    <div v-if="showDemoImages">
      <div class="demoimages">
        <div class="demoimage">
          <div style="color: rgb(70, 122, 253);">Your Content</div>
          <img :src="sampleImage" style="max-width: 400px;" />
        </div>
        <div class="demoimage">
          <div style="color: rgb(70, 122, 253);">Your Result</div>
          <img :src="sampleImageResult" style="max-width: 400px;" />
        </div>
      </div>
    </div>
  </div>
  
  <div class = "uploadcontainer">
    <div class="upload" @dragover="onDragOver" @dragleave="onDragLeave" @drop="onDrop" :class="{ 'drag-over': dragOver }">
      <input type="file" ref="fileInput" accept="image/png" @change="onFileChange" style="display: none;" />
      <button class="button" @click="selectFile">Upload Image</button>
      <p>Or drag and drop an image here</p>
      <div v-if="uploadedFile">
        <button class="button" @click="submitFile">Submit Image</button>
        <button class="button" @click="deleteFile">Delete Image</button>
        <p>You've uploaded file: {{ uploadedFileName }}</p>
      </div>
    </div>
  </div>

  <div v-if="grayscaleImageUrl" class="grayscale-container">
    <h3>Your Analyzed Image</h3>
    <img :src="grayscaleImageUrl" alt="Your Image" style="max-width: 400px;" />
  </div>
  
</template>

<style scoped>
* {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

button {
  background-color: white;
  color: rgb(70, 122, 253);
  font-size: large;
  padding: 10px;
  justify-content: center;
}

.demo {
  display: flex;
  justify-content: center;
  align-items: center;
  padding-top: 60px;
}

.upload-container {
  display: flex;
  justify-content: center;
  align-items: center;
  padding-top: 100px;
}

.upload {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding-top: 100px;
  border: 2px dashed rgb(70, 122, 253);
  padding: 20px;
  text-align: center;
  cursor: pointer;
  background-color: rgb(146, 177, 255);
  color: white;
  width: 300px;
  margin:auto
}



.upload.drag-over {
  background-color: rgba(70, 122, 253, 0.8);
}

button:hover {
  background-color: rgb(70, 122, 253);
  color: white;
}

.demoimages {
  display: flex;
  justify-content: center;
  padding: 25px;
}
</style>
