export const predictHeartDisease = async (patientData) => {
  const response = await fetch('https://heart-disease-prediction-api-production.up.railway.app/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: 'Hybrid GA-PSO-ANN',
      patient: patientData,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    console.error('Prediction API error:', error);
    throw new Error(error.detail || 'Prediction failed. Please check your inputs and try again.');
  }

  return response.json();
};
