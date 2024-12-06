# EmoCare: Digital Emotion Assistant for Mental Health

## Project Description
**EmoCare** is a web-based application designed to analyze and predict emotions from text inputs. This application uses a **BERT-based model** to classify six emotional states from the provided text: **Caring**, **Love**, **Gratitude**, **Sadness**, **Fear**, and **Anger**. The app aims to assist individuals in understanding and managing their emotions, contributing to mental well-being. EmoCare also provides personalized suggestions based on the predicted emotions, leveraging **OpenAI GPT** for generating supportive suggestions.

### Features
- **Emotion Prediction**: Predicts six types of emotions (Caring, Love, Gratitude, Sadness, Fear, and Anger) from text input.
- **Emotion Suggestions**: Provides actionable mental well-being suggestions based on predicted emotions.
- **Real-Time Processing**: The model processes text and returns emotion predictions within seconds.
- **Dynamic Backend**: Built with **FastAPI** for a fast, scalable API.
- **Easy Integration**: Designed for easy integration with web-based frontend frameworks.

### Technologies Used
- **Backend**: FastAPI
- **Frontend**: React.js (hosted on Vercel)
- **Emotion Detection Model**: BERT (HuggingFace)
- **Text Generation**: OpenAI GPT-4
- **Environment**: Python 3.x, FastAPI, Pytorch, Transformers, OpenAI API
