<script setup>
import { ref } from 'vue';
import axios from 'axios';

const sampleImage = 'src/images/sample.png';
const sampleImageResult = 'src/images/udlmock1result.png';

const showDemoImages = ref(false);
const uploadedFile = ref(null);
const uploadedFileName = ref('');
const showResponse = ref('');
const fileInput = ref(null);
const dragOver = ref(false);

const grayscaleImageUrl = ref('');


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
    const response = await axios.post('http://localhost:8000/upload/', formData, {
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
