"use client"

import { useState, useRef } from "react"
import ChatMessage from "@/components/chat-message"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

import {
  GitBranch,
  Wrench,
  Users,
  Paperclip,
  X,
  PanelLeftClose,
  PanelLeftOpen,
  Image,
  Database
} from "lucide-react"

export default function Home() {

  const [messages, setMessages] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([])

  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [input, setInput] = useState("")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedRag, setSelectedRag] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement | null>(null)

  function toggleRag(mode: string) {
    setSelectedRag(selectedRag === mode ? null : mode)
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {

    const file = e.target.files?.[0]
    if (!file) return

    // SHOW FILE IMMEDIATELY
    setUploadedFiles(prev => [...prev, file.name])

    const formData = new FormData()
    formData.append("file", file)

    try {

      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
      })

      const data = await response.json()

      // SHOW MESSAGE AFTER PROCESSING
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: data.message }
      ])

    } catch {

      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Error processing file." }
      ])

    }
  }

  function removeFile(index: number) {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  function sendMessage() {

    if (!input.trim()) return

    setMessages(prev => [
      ...prev,
      { role: "user", content: input },
      { role: "assistant", content: "Thinking..." }
    ])

    setInput("")
  }

  const buttonBase =
    "w-full justify-start text-2xl py-8 px-6 transition-all duration-200 flex items-center gap-6"

  const hoverStyle =
    "bg-white/10 text-white hover:bg-white/40 hover:text-black hover:scale-[1.02]"

  const activeStyle =
    "bg-white text-black scale-[1.02]"

  return (
    <div className="flex h-screen w-full overflow-hidden">

      {/* Sidebar */}

      <aside
        className={`border-r border-white/10 bg-gradient-to-b from-indigo-600 via-purple-600 to-blue-600 p-6 flex flex-col gap-6 transition-all duration-300 ${
          sidebarOpen ? "w-[440px]" : "w-20"
        }`}
      >

        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="text-white/80 hover:text-white flex justify-end"
        >
          {sidebarOpen ? <PanelLeftClose size={34} /> : <PanelLeftOpen size={34} />}
        </button>

        {sidebarOpen && (
          <h2 className="text-4xl font-bold text-white mb-4">
            RAG Agents
          </h2>
        )}

        <Button
          onClick={() => toggleRag("adaptive")}
          className={`${buttonBase} ${
            selectedRag === "adaptive" ? activeStyle : hoverStyle
          }`}
        >
          <GitBranch size={52} />
          {sidebarOpen && "Adaptive RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("corrective")}
          className={`${buttonBase} ${
            selectedRag === "corrective" ? activeStyle : hoverStyle
          }`}
        >
          <Wrench size={52} />
          {sidebarOpen && "Corrective RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("multi")}
          className={`${buttonBase} ${
            selectedRag === "multi" ? activeStyle : hoverStyle
          }`}
        >
          <Users size={52} />
          {sidebarOpen && "Multi-Agent"}
        </Button>

        <Button
          onClick={() => toggleRag("multimodal")}
          className={`${buttonBase} ${
            selectedRag === "multimodal" ? activeStyle : hoverStyle
          }`}
        >
          <Image size={52} />
          {sidebarOpen && "Multi-Modal RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("cache")}
          className={`${buttonBase} ${
            selectedRag === "cache" ? activeStyle : hoverStyle
          }`}
        >
          <Database size={52} />
          {sidebarOpen && "Cache RAG"}
        </Button>

        {/* Upload */}

        <div className="mt-6 flex flex-col gap-4">

          <input
            type="file"
            accept=".txt"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
          />

          <Button
            onClick={() => fileInputRef.current?.click()}
            className="w-full justify-start text-2xl py-7 px-6 bg-white/10 text-white hover:bg-white/40 hover:text-black hover:scale-[1.02] transition-all duration-200 flex items-center gap-6"
          >
            <Paperclip size={50} />
            {sidebarOpen && "Upload File"}
          </Button>

          {sidebarOpen && (
            <div className="text-lg text-white mt-3 space-y-3">

              {uploadedFiles.length === 0 && (
                <p className="text-white/50">No files uploaded</p>
              )}

              {uploadedFiles.map((file, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-white/15 rounded-xl px-4 py-3"
                >
                  <span>{file}</span>

                  <button
                    onClick={() => removeFile(i)}
                    className="text-white/80 hover:text-red-400"
                  >
                    <X size={26} />
                  </button>

                </div>
              ))}

            </div>
          )}

        </div>

      </aside>

      {/* Chat */}

      <main className="flex min-w-0 flex-1 flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">

        <ScrollArea className="flex-1">

          {messages.length === 0 ? (

            <div className="h-full flex items-center justify-center text-white/70">
              Upload a file to build the vector store.
            </div>

          ) : (

            <div className="mx-auto max-w-3xl space-y-6 p-8">

              {messages.map((m, i) => (
                <ChatMessage
                  key={i}
                  role={m.role}
                  content={m.content}
                />
              ))}

            </div>

          )}

        </ScrollArea>

        <div className="border-t border-white/10 bg-black/20 backdrop-blur p-6">

          <div className="mx-auto flex max-w-3xl gap-3 items-end">

            <Textarea
              placeholder="Ask something..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="min-h-[60px] border-white/10 bg-white/5 text-white"
            />

            <Button
              onClick={sendMessage}
              className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white"
            >
              Send
            </Button>

          </div>

        </div>

      </main>

    </div>
  )
}