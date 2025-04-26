<script setup>
import { ref, onMounted, computed, nextTick} from 'vue';
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

const showlatex = ref(false);
const latexResults = ref([]);

const showProcessedImage = ref(false);
const processedImage = ref('');
const currentTaskId = ref(null);
const imHeight = ref(0);
const imWidth = ref(0);
const scaledHeight = ref(0);
const scaledWidth = ref(0);

function toggleResponse(msg) {
  showResponse.value = msg;
}

function toggleProcessedImages() {
    showProcessedImage.value = !showProcessedImage.value;
}

function toggleLatex() {
  showlatex.value = !showlatex.value;
}

function selectFile() {
  fileInput.value.click();
}

const isProcessing = computed(() => currentTaskId.value !== null);

function onFileChange(event) {
  if (isProcessing.value) {
    toggleResponse('Please wait for current processing to complete');
    event.target.value = ''; // Reset the file input
    return;
  }
  
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
  
  if (isProcessing.value) {
    toggleResponse('Please wait for current processing to complete');
    return;
  }
  
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

async function fetchVisualization(taskId) {
  try {
    const response = await fetch(`${import.meta.env.VITE_ROOT_API}/task/${taskId}/image`);
    if (!response.ok) {
      console.error('Failed to fetch visualization');
      return;
    }
    const blob = await response.blob();
    processedImage.value = URL.createObjectURL(blob);
    showProcessedImage.value = true;
    const im = new Image();
    im.src = processedImage.value;
    im.onload = () => {
      imHeight.value = im.height;
      imWidth.value = im.width;

      nextTick(() => {
        const htmlImage = document.getElementById("processedImageResize");
        if (htmlImage) {
          scaledWidth.value = htmlImage.clientWidth;
          scaledHeight.value = htmlImage.clientHeight;
        }
      });
    };
  } catch (error) {
    console.error('Error fetching visualization:', error);
  }
}

function resetState() {
  uploadedFile.value = null;
  uploadedFileName.value = '';
  fileInput.value.value = '';
  processedImage.value = '';
  showProcessedImage.value = false;
  showlatex.value = false;
  currentTaskId.value = null;
  console.log('State reset complete');
}

async function submitFile() {
  if (!uploadedFile.value) {
    toggleResponse('No file selected');
    return;
  }
  
  if (isProcessing.value) {
    toggleResponse('Please wait for current processing to complete');
    return;
  }

  const formData = new FormData();
  formData.append('file', uploadedFile.value);
  
  try {
    // Upload the file
    const response = await fetch(`${import.meta.env.VITE_ROOT_API}/upload`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const data = await response.json();
      console.error(data.message);
      toggleResponse(data.message);
      resetState();
      return;
    }

    const { task_id, message } = await response.json();
    currentTaskId.value = task_id;
    toggleResponse("Processing your image...");

    // Start SSE connection
    const eventSource = new EventSource(`${import.meta.env.VITE_ROOT_API}/task/${task_id}`);

    eventSource.onmessage = (event) => {
      console.log('Received event:', event);
    };

    eventSource.addEventListener('result', async (event) => {
      const results = JSON.parse(event.data);
      console.log('Received results:', results);
      latexResults.value = results.map(result => ({
        task: result.cls,
        bbox: result.bbox,
        text: result.text
      }));
      toggleResponse("Processing complete!");
      showlatex.value = true;
      
      // Fetch visualization after receiving results
      await fetchVisualization(task_id);
      
      // Reset task ID to allow new uploads
      currentTaskId.value = null;
    });

    eventSource.addEventListener('error', (event) => {
      console.error('SSE Error:', event);
      toggleResponse("Error processing image");
      eventSource.close();
      resetState();
    });

    eventSource.addEventListener('close', () => {
      console.log('SSE connection closed');
      eventSource.close();
      // Don't reset state here as we want to keep the results visible
      currentTaskId.value = null;
    });

  } catch (error) {
    console.error(error);
    toggleResponse("Couldn't upload image");
    resetState();
  }
}

const formula = computed(() => {
  return latexResults.value.map(element => {
    const f = element.text;
    if (element.task === "text") {
        return f;
    }
    return katex.renderToString(f, { throwOnError: false });
  });
})

function toggleImages() {
  showDemoImages.value = !showDemoImages.value;
}

function deleteFile() {
  if (uploadedFile.value) {
    resetState();
    toggleResponse('');
    console.log('File deleted and state reset');
  }
}

const uploadedImageURL = computed(() => {
  return uploadedFile.value ? URL.createObjectURL(uploadedFile.value) : null;
});

async function copyLatex(text) {
  try {
    await navigator.clipboard.writeText(text);
    alert("Text/Latex copied: "+text);
    console.log("latex copied");
    console.log(imHeight.value+" "+imWidth.value+" "+scaledHeight.value+" "+imWidth.value);
  } catch (err) {
    console.error("Error:", err);
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
  
  <div class="uploadcontainer">
    <div class="upload" @dragover="onDragOver" @dragleave="onDragLeave" @drop="onDrop" 
         :class="{ 'drag-over': dragOver, 'processing': isProcessing }">
      <input type="file" ref="fileInput" accept="image/png" @change="onFileChange" 
             style="display: none;" :disabled="isProcessing" />
      <button class="button" @click="selectFile" :disabled="isProcessing">
        Upload Image
      </button>
      <p>Or drag and drop an image here</p>
      <div v-if="uploadedFile">
        <button class="button" @click="submitFile" :disabled="isProcessing">
          {{ isProcessing ? 'Processing...' : 'Submit Image' }}
        </button>
        <button class="button" @click="deleteFile" :disabled="isProcessing">
          Delete Image
        </button>
        <p>You've uploaded file: {{ uploadedFileName }}</p>
      </div>
    </div>
  </div>

  <div v-if="showProcessedImage || showlatex" class="results-container">
    <div v-if="showProcessedImage" class="visualization-container">
      <h3>Layout Detection</h3>
      <div class = "overlay">
        <img :src="uploadedImageURL" alt="Uploaded image preview" class="visualization-image" id="uploadedImageResize" />
        <!--  -->
        <div v-for="(item, ind) in latexResults" v-if="imHeight && imWidth" :key="ind">
          <button
            @click="copyLatex(item.text)"
            :style="{
              position: 'absolute',
              left: (((item.bbox[0] / imWidth) * 100)-2) + '%',
              top: (((item.bbox[1] / imHeight) * 100)-2) + '%',
              width: (((item.bbox[2] - item.bbox[0]) / imWidth * 100)+4) + '%',
              height: (((item.bbox[3] - item.bbox[1]) / imHeight * 100)+4) + '%',
              border: '2px solid rgb(70, 122, 253)',
              backgroundColor: 'transparent',
              cursor: 'pointer',
            }"
          >
          </button>
        </div>
      </div>
      <img :src="processedImage" alt="Processed visualization" class="visualization-image" id="processedImageResize" />
    </div>

    <div v-if="showlatex" class="formulas-container">
      <h3>Detected Formulas</h3>
      <ol>
        <li v-for="(item, ind) in latexResults" :key="ind">
          <ul>
            <li>Type: {{ item.task }}</li>
            <li>Position: {{ item.bbox.join(', ') }}</li>
            <li v-html="formula[ind]" @click="copyLatex(item.text)"></li>
          </ul>
        </li>
      </ol>
    </div>
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

.overlay {
  position: relative;
  display: inline-block;
}

/* .overlaybox {
  position:absolute;
} */

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
  margin: auto;
}

.upload.drag-over {
  background-color: rgba(70, 122, 253, 0.8);
}

.upload.processing {
  opacity: 0.7;
  cursor: not-allowed;
}

button:hover {
  background-color: rgb(70, 122, 253);
  color: white;
}

button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  background-color: #ccc;
  color: #666;
}

button:disabled:hover {
  background-color: #ccc;
  color: #666;
}

.demoimages {
  display: flex;
  justify-content: center;
  padding: 25px;
}

.results-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2rem;
  padding: 2rem;
}

.visualization-container {
  text-align: center;
  max-width: 800px;
  width: 100%;
}

.visualization-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.formulas-container {
  max-width: 800px;
  width: 100%;
}

.formulas-container ol {
  list-style-type: decimal;
  padding-left: 1.5rem;
}

.formulas-container ul {
  list-style-type: none;
  padding-left: 0;
}

.formulas-container li {
  margin-bottom: 1rem;
}
</style>
