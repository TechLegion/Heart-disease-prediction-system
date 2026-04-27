# CardioSense — Setup Guide

## Prerequisites
- Node.js 18+
- A Firebase project (free Spark plan works)

## 1. Clone & Install
```bash
git clone <your-repo>
cd cardiosense
npm install
```

## 2. Firebase Setup
1. Go to https://console.firebase.google.com
2. Create a new project named "CardioSense"
3. Enable **Authentication** -> Sign-in method -> **Email/Password** -> Enable
4. Enable **Firestore Database** -> Create database -> Start in **production mode**
5. Go to Project Settings -> Your apps -> Add a **Web app**
6. Copy the config object

## 3. Environment Variables
Create a `.env` file in the project root:
```env
VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id
```

## 4. Firestore Security Rules
In Firebase Console -> Firestore -> Rules, paste:
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /patients/{patientId} {
      allow read, write: if request.auth != null
        && request.auth.uid == resource.data.uid;
      allow create: if request.auth != null;
    }
  }
}
```
Click **Publish**.

## 5. Run Locally
```bash
npm run dev
```
Open http://localhost:5173

## 6. Build for Production
```bash
npm run build
npm run preview
```

## API Reference
The app calls:
`POST https://heart-disease-prediction-api-production.up.railway.app/api/predict`
No API key required. Internet connection needed.

## Troubleshooting
- **CORS error on API call**: The API must be called from the browser (client-side fetch), not from a server/Node environment.
- **Firebase permission denied**: Check that your Firestore rules are published and that the user is logged in.
- **Blank dashboard**: Create your first prediction via the "New Prediction" page.
