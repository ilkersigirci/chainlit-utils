import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"

export default function HumanReview() {
  const reviews = Array.isArray(props.reviews) ? props.reviews : []
  const control = props._chainlit_utils_hitl || {}
  const [responses, setResponses] = useState(() => reviews.map(() => ""))
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  const setResponse = (index, value) => {
    setResponses((current) =>
      reviews.map((_, reviewIndex) =>
        reviewIndex === index ? value : current[reviewIndex] || ""
      )
    )
  }

  const controlReady =
    typeof control.action === "string" &&
    typeof control.step_id === "string" &&
    typeof control.element_id === "string" &&
    typeof control.revision === "string"
  const ready =
    controlReady &&
    reviews.length > 0 &&
    responses.length === reviews.length &&
    responses.every((response) => response.trim())

  // The workflow records the review and removes this form before it replies.
  const send = async (decision) => {
    if (!controlReady || submitting) return
    setSubmitting(true)
    setError("")
    try {
      const result = await callAction({
        name: control.action,
        payload: {
          step_id: control.step_id,
          element_id: control.element_id,
          revision: control.revision,
          ...decision,
        },
      })
      if (result?.response?.ok !== true) {
        setError(
          result?.response?.error || "The human review could not be submitted."
        )
      }
    } catch (submissionError) {
      setError(String(submissionError))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card className="w-full max-w-xl">
      <CardHeader>
        <CardTitle className="text-base">Human review</CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {reviews.map((review, index) => {
          const choices = Array.isArray(review.choices) ? review.choices : []
          const allowOther = review.allow_other === true || choices.length === 0
          const response = responses[index] || ""
          const inputId = `human-review-response-${index}`

          return (
            <section key={index} className="space-y-3">
              {/* The message above lists the prompts; numbers pair each with its answer. */}
              {reviews.length > 1 && (
                <p className="text-sm font-medium">Request {index + 1}</p>
              )}

              {choices.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {choices.map((choice) => (
                    <Button
                      key={choice}
                      type="button"
                      variant={response === choice ? "default" : "outline"}
                      disabled={submitting}
                      onClick={() => setResponse(index, choice)}
                    >
                      {choice}
                    </Button>
                  ))}
                </div>
              )}

              {allowOther && (
                <div className="space-y-2">
                  <Label htmlFor={inputId}>Custom response</Label>
                  <Textarea
                    id={inputId}
                    placeholder="Write a response..."
                    rows={3}
                    value={response}
                    disabled={submitting}
                    onChange={(event) => setResponse(index, event.target.value)}
                  />
                </div>
              )}
            </section>
          )
        })}

        {error && <p className="text-sm text-destructive">{error}</p>}
      </CardContent>

      <CardFooter className="justify-end gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={() => send({ cancel: true })}
          disabled={!controlReady || submitting}
        >
          Cancel
        </Button>
        <Button
          type="button"
          onClick={() =>
            send({ outputs: responses.map((response) => response.trim()) })
          }
          disabled={!ready || submitting}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  )
}
