import { Avatar, AvatarFallback } from "@/components/ui/avatar"

interface Props {
  role: "user" | "assistant" | "system" | "general"
  content: string
}

export default function ChatMessage({ role, content }: Props) {

  // system messages appear centered in blue
  if (role === "system") {
    return (
      <div className="flex justify-center">
        <div
          className={`
          max-w-2xl rounded-2xl px-6 py-4 border shadow-xl backdrop-blur
          text-lg leading-relaxed text-center
          bg-gradient-to-r from-blue-600 to-indigo-700
          text-blue-50
          border-blue-400/30
          shadow-blue-900/30
          font-semibold
          `}
        >
          {content}
        </div>
      </div>
    )
  }

  // general messages (file processing, status messages) appear centered in gold
  if (role === "general") {
    return (
      <div className="flex justify-center">
        <div
          className={`
          max-w-2xl rounded-2xl px-6 py-4 border shadow-xl backdrop-blur
          text-lg leading-relaxed text-center
          bg-gradient-to-r from-amber-500 to-yellow-600
          text-amber-50
          border-amber-300/30
          shadow-amber-900/30
          font-semibold
          `}
        >
          {content}
        </div>
      </div>
    )
  }

  return (
    <div className={`flex gap-4 ${role === "user" ? "justify-end" : ""}`}>

      {role === "assistant" && (
        <Avatar>
          <AvatarFallback className="bg-green-700 text-green-100 font-bold">
            AI
          </AvatarFallback>
        </Avatar>
      )}

      <div
        className={`max-w-2xl rounded-2xl px-6 py-4 border shadow-xl backdrop-blur text-lg leading-relaxed ${
          role === "user"
            ? `
            bg-gradient-to-r from-red-600 to-rose-700
            text-red-50
            border-red-400/30
            shadow-red-900/30
            font-semibold
            `
            : `
            bg-gradient-to-r from-green-600 to-emerald-700
            text-green-50
            border-green-400/30
            shadow-green-900/30
            font-medium
            `
        }`}
      >
        {content}
      </div>

      {role === "user" && (
        <Avatar>
          <AvatarFallback className="bg-red-700 text-red-100 font-bold">
            U
          </AvatarFallback>
        </Avatar>
      )}

    </div>
  )
}