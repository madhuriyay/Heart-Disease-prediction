async function predict() {
    const age = document.getElementById("age").value;
    const chol = document.getElementById("chol").value;
    // Add other inputs

    const features = [age, chol /* ... other features in same order */];

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: features })
    });

    const data = await response.json();
    document.getElementById("result").innerText = "Prediction: " + (data.prediction === 1 ? "Heart Disease" : "No Disease");
}