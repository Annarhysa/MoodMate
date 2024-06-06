document.getElementById('ask-btn').addEventListener('click', async () => {
    const question = document.getElementById('question').value;
    try {
        const response = await fetch('http://127.0.0.1:5000/api/get_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById('answer').innerText = data.answer;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('answer').innerText = 'An error occurred. Please try again.';
    }
});
