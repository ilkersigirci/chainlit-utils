import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Loader2, Square, Volume2 } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"

export default function SpeechButton() {
  const clips = useRef([])
  const current = useRef(null)
  const stopped = useRef(false)
  const [state, setState] = useState("idle")

  // Fetched parts stay cached, so replaying the answer needs no new speech.
  const load = async (part) => {
    if (!clips.current[part]) {
      const result = await callAction({
        name: props.action,
        payload: { message_id: props.message_id, part },
      })
      const response = result?.response
      if (response?.ok !== true || typeof response.audio !== "string") {
        throw new Error(response?.error || "Speech failed.")
      }
      clips.current[part] = {
        audio: new Audio(`data:audio/mpeg;base64,${response.audio}`),
        more: response.more === true,
      }
    }
    return clips.current[part]
  }

  const play = (audio) =>
    new Promise((resolve, reject) => {
      if (stopped.current) return resolve()
      current.current = audio
      audio.currentTime = 0
      audio.onended = audio.onpause = resolve
      setState("playing")
      audio.play().catch((error) => {
        stopped.current = true
        reject(error)
      })
    })

  const stop = () => {
    stopped.current = true
    current.current?.pause()
  }

  const toggle = async () => {
    if (state !== "idle") return stop()
    stopped.current = false
    setState("loading")
    // Like Open WebUI: synthesize one part at a time and play them in order.
    let queue = Promise.resolve()
    try {
      for (let part = 0, more = true; more && !stopped.current; part++) {
        const clip = await load(part)
        more = clip.more
        queue = queue.then(() => play(clip.audio))
      }
      await queue
    } catch (error) {
      stop()
      toast.error(error?.message || "Speech failed.")
    }
    setState("idle")
  }

  useEffect(() => stop, [])

  const label = state === "idle" ? "Read aloud" : "Stop"
  return (
    <div className="-ml-1.5 flex items-center">
      <TooltipProvider delayDuration={100}>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              aria-label={label}
              className="text-muted-foreground"
              onClick={toggle}
              size="icon"
              variant="ghost"
            >
              {state === "loading" ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : state === "playing" ? (
                <Square className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{label}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  )
}
