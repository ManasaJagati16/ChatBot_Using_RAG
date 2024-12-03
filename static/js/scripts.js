const inputForm = document.getElementById('input-form');
const inputField = document.getElementById('input-field');
const fileInput = document.getElementById('file-upload');
const conversation = document.getElementById('conversation');
const submitButton = document.getElementById('submit-button');
const spinnerContainer = document.getElementById('spinner-container');

inputForm.addEventListener('submit', async function (event) {
    event.preventDefault();

    const userMessage = inputField.value.trim();
    const file = fileInput.files[0];
    const summaryType = document.getElementById('summary-type').value;
    const section = document.getElementById('section').value;

    if (!userMessage && !file) return;

    if (userMessage || file) addMessage(inputField.value, 'user-message');

    const formData = new FormData();
    if (userMessage && !file) formData.append('message', userMessage);
    if (file) formData.append('file', file);
    formData.append('summary_type', summaryType);
    formData.append('section', section);

    showSpinner();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        hideSpinner();

        if (data.response) {
            addMessage(data.response, 'chatbot-message');

            // Show download link
            const downloadContainer = document.getElementById('download-container');
            const downloadLink = document.getElementById('download-link');
            downloadLink.href = '/download_summary';
            downloadContainer.style.display = 'block';
        } else if (data.error) {
            addMessage(`Error: ${data.error}`, 'chatbot-message');
        }
    } catch (error) {
        hideSpinner();
        addMessage('Error connecting to the server. Please try again later.', 'chatbot-message');
        console.error('API Error:', error);
    }

    fileInput.value = '';
});

function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add(type);

    const messageText = document.createElement('p');
    messageText.classList.add(type === 'user-message' ? 'user-text' : 'chatbot-text');
    messageText.textContent = content;

    messageDiv.appendChild(messageText);
    conversation.appendChild(messageDiv);

    // Scroll to the latest message
    messageDiv.scrollIntoView({ behavior: 'smooth' });
}

function showSpinner() {
    spinnerContainer.style.display = 'block';
    submitButton.disabled = true;
}

function hideSpinner() {
    spinnerContainer.style.display = 'none';
    submitButton.disabled = false;
}

fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file) {
        // Display filename in input field
        inputField.value = file.name;
        inputField.disabled = true;
        inputField.placeholder = `Uploaded: ${file.name}`;
    }
});
