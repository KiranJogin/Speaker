const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const loading = document.getElementById("loading");
const output = document.getElementById("output");
const scriptOutput = document.getElementById("script-output");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) return alert("Please select an audio file.");

  loading.classList.remove("hidden");
  output.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/transcribe", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
    } else {
      scriptOutput.textContent = data.formatted_script;
      output.classList.remove("hidden");
    }
  } catch (err) {
    alert("Request failed: " + err.message);
  } finally {
    loading.classList.add("hidden");
  }
});
    