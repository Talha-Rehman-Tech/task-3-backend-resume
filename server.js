import express from 'express'
import { GoogleGenerativeAI } from '@google/generative-ai'
import dotenv from 'dotenv'
import cors from 'cors'

dotenv.config()

const app = express()
app.use(cors())
const port = process.env.PORT || 3000
const apiKey = process.env.GEMINI_API_KEY
let genAI = null

if (apiKey) {
  genAI = new GoogleGenerativeAI(apiKey)
}

const getGenerativeClient = () => {
  if (!genAI) {
    throw new Error('Missing GEMINI_API_KEY environment variable.')
  }
  return genAI
}

const cleanModelText = (text) => {
  let cleaned = text.trim()

  if (cleaned.startsWith('```json')) {
    cleaned = cleaned.substring(7)
  }
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.substring(3)
  }
  if (cleaned.endsWith('```')) {
    cleaned = cleaned.substring(0, cleaned.length - 3)
  }

  return cleaned.trim()
}

app.use(express.json())

app.get('/', (_req, res) => {
  res.json({ status: 'ok', message: 'Resume analyzer backend is running.' })
})

app.post('/api/analyze', async (req, res) => {
  const { resumeText, jobDescription } = req.body

  if (!resumeText) {
    return res.status(400).json({ error: 'Resume text is required.' })
  }

  try {
    const client = getGenerativeClient()
    const model = client.getGenerativeModel({ model: 'gemini-flash-lite-latest' })
    const prompt = `
You are an expert HR professional and ATS (Applicant Tracking System) algorithm.
Analyze the following resume against the provided target job description.

Resume:
${resumeText}

Job Description:
${jobDescription || 'Not provided (evaluate general resume quality instead).'}

Return ONLY a valid JSON object with the following structure (no markdown blocks, no backticks, just the raw JSON):
{
  "score": <number between 0 and 100 representing overall quality/match>,
  "matchRate": <number between 0 and 100 representing keyword match percentage>,
  "strengths": [<array of 3-5 strings detailing strong points>],
  "weaknesses": [<array of 3-5 strings detailing areas for improvement>]
}
`

    try {
      const result = await model.generateContent(prompt)
      const response = await result.response
      const text = cleanModelText(response.text())
      const parsed = JSON.parse(text)
      return res.json(parsed)
    } catch (error) {
      console.error('Error calling Gemini API:', error)
      return res.status(500).json({ error: 'Failed to analyze resume. Please try again.' })
    }
  } catch (error) {
    console.error('Missing Gemini configuration:', error.message)
    return res.status(500).json({ error: 'Server is not configured with GEMINI_API_KEY.' })
  }
})

app.post('/api/chat', async (req, res) => {
  const { message, chatHistory = [], resumeText = '', jobDescription = '' } = req.body

  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'Message is required.' })
  }

  try {
    const client = getGenerativeClient()
    const systemContext = `
You are a helpful career coach and resume expert.
The user has uploaded their resume and optionally a target job description.

Here is their resume:
${resumeText || 'No resume uploaded yet.'}

Target Job Description:
${jobDescription || 'Not provided.'}

Answer their questions specifically based on their resume to help them improve it or prepare for an interview. Keep responses concise and actionable.
`
    const model = client.getGenerativeModel({
      model: 'gemini-flash-lite-latest',
      systemInstruction: { parts: [{ text: systemContext }] }
    })

    try {
      const formattedHistory = chatHistory.map((msg) => ({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.text }]
      }))

      const chat = model.startChat({
        history: formattedHistory
      })

      const result = await chat.sendMessage(message)
      const response = await result.response
      return res.json({ text: response.text() })
    } catch (error) {
      console.error('Error in AI chat:', error)
      return res.status(500).json({ error: 'Failed to get response. Please try again.' })
    }
  } catch (error) {
    console.error('Missing Gemini configuration:', error.message)
    return res.status(500).json({ error: 'Server is not configured with GEMINI_API_KEY.' })
  }
})

const startServer = () => {
  if (process.env.NODE_ENV === 'production') {
    return
  }

  const server = app.listen(port, () => {
    console.log(`Server listening on http://localhost:${port}`)
  })

  server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
      console.error(`Port ${port} is already in use. Stop the other server or use a different port.`)
    } else {
      console.error('Server startup error:', error)
    }
  })
}

startServer()

export default app

