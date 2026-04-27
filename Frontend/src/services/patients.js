import { addDoc, collection, deleteDoc, doc, getDoc, getDocs, query, serverTimestamp, where } from 'firebase/firestore';
import { db } from './firebase';

const patientsCollection = collection(db, 'patients');

export const createPatientRecord = async (record) => {
  const docRef = await addDoc(patientsCollection, {
    ...record,
    createdAt: serverTimestamp(),
  });

  return docRef.id;
};

export const fetchPatients = async (uid) => {
  const q = query(patientsCollection, where('uid', '==', uid));
  const snapshot = await getDocs(q);

  return snapshot.docs
    .map((document) => ({ id: document.id, ...document.data() }))
    .sort((first, second) => {
      const firstTime = first.createdAt?.toMillis?.() ?? 0;
      const secondTime = second.createdAt?.toMillis?.() ?? 0;
      return secondTime - firstTime;
    });
};

export const fetchPatientById = async (id) => {
  const snapshot = await getDoc(doc(db, 'patients', id));
  if (!snapshot.exists()) {
    return null;
  }
  return { id: snapshot.id, ...snapshot.data() };
};

export const deletePatientById = async (id) => {
  await deleteDoc(doc(db, 'patients', id));
};
