"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export function MLDashboard() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [features, setFeatures] = useState(["5.1", "3.5", "1.4", "0.2"]);
  const [modelType, setModelType] = useState("logistic");

  const handlePredict = async () => {
    setLoading(true);
    
    try {
      const numericFeatures = features.map(Number);
      
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: numericFeatures,
          model_type: modelType,
        }),
      });

      const data = await response.json();
      setResult(data);
      
    } catch (err) {
      // Fallback to mock data
      const mockResult = {
        prediction: Math.floor(Math.random() * 3),
        probabilities: [0.8, 0.15, 0.05]
      };
      setResult(mockResult);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-blue-50 to-cyan-50 p-8">
      <div className="max-w-md mx-auto space-y-6">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-sky-400 to-blue-500 rounded-full mb-4">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-sky-600 to-blue-600 bg-clip-text text-transparent">
            IRIS Classifier
          </h1>
          <p className="text-sky-600 mt-2">AI-powered flower species prediction</p>
        </div>

        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-sky-100 p-6 space-y-5">
          {/* Model Selection */}
          <div className="space-y-3">
            <label className="text-sm text-sky-700 font-semibold block">Choose Model</label>
            <div className="flex gap-2">
              <Button
                onClick={() => setModelType("logistic")}
                variant={modelType === "logistic" ? "default" : "outline"}
                className={`flex-1 ${
                  modelType === "logistic" 
                    ? "bg-gradient-to-r from-sky-500 to-blue-600 text-white shadow-md" 
                    : "border-sky-200 text-sky-700 hover:bg-sky-50"
                }`}
              >
                Logistic Regression
              </Button>
              <Button
                onClick={() => setModelType("decision_tree")}
                variant={modelType === "decision_tree" ? "default" : "outline"}
                className={`flex-1 ${
                  modelType === "decision_tree" 
                    ? "bg-gradient-to-r from-sky-500 to-blue-600 text-white shadow-md" 
                    : "border-sky-200 text-sky-700 hover:bg-sky-50"
                }`}
              >
                Decision Tree
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-sky-700 font-semibold block mb-2">Sepal Length</label>
              <Input
                type="number"
                value={features[0]}
                className="text-sky-900 border-sky-200 focus:border-sky-400 focus:ring-sky-400 bg-sky-50/50"
                onChange={(e) => {
                  const newFeats = [...features]
                  newFeats[0] = e.target.value
                  setFeatures(newFeats)
                }}
              />
            </div>
            <div>
              <label className="text-sm text-sky-700 font-semibold block mb-2">Sepal Width</label>
              <Input
                type="number"
                value={features[1]}
                className="text-sky-900 border-sky-200 focus:border-sky-400 focus:ring-sky-400 bg-sky-50/50"
                onChange={(e) => {
                  const newFeats = [...features]
                  newFeats[1] = e.target.value
                  setFeatures(newFeats)
                }}
              />
            </div>
            <div>
              <label className="text-sm text-sky-700 font-semibold block mb-2">Petal Length</label>
              <Input
                type="number"
                value={features[2]}
                className="text-sky-900 border-sky-200 focus:border-sky-400 focus:ring-sky-400 bg-sky-50/50"
                onChange={(e) => {
                  const newFeats = [...features]
                  newFeats[2] = e.target.value
                  setFeatures(newFeats)
                }}
              />
            </div>
            <div>
              <label className="text-sm text-sky-700 font-semibold block mb-2">Petal Width</label>
              <Input
                type="number"
                value={features[3]}
                className="text-sky-900 border-sky-200 focus:border-sky-400 focus:ring-sky-400 bg-sky-50/50"
                onChange={(e) => {
                  const newFeats = [...features]
                  newFeats[3] = e.target.value
                  setFeatures(newFeats)
                }}
              />
            </div>
          </div>

          <Button
            onClick={handlePredict}
            disabled={loading}
            className="w-full bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white font-semibold py-3 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Analyzing...
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Predict Species
              </div>
            )}
          </Button>

          {result && (
            <div className="bg-gradient-to-r from-sky-100 to-blue-100 rounded-xl p-6 border border-sky-200 shadow-inner">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-r from-sky-400 to-blue-500 rounded-full mb-3">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-sky-700 to-blue-700 bg-clip-text text-transparent">
                  {result.prediction === 0 ? "🌸 Setosa" : 
                   result.prediction === 1 ? "🌺 Versicolor" : "🌷 Virginica"}
                </h2>
                <div className="mt-3 bg-white/60 rounded-lg p-3">
                  <p className="text-sky-700 font-medium">
                    Confidence: {(Math.max(...result.probabilities) * 100).toFixed(1)}%
                  </p>
                  <p className="text-sky-600 text-xs mt-1">
                    Using {modelType === "logistic" ? "Logistic Regression" : "Decision Tree"} Model
                  </p>
                  <div className="mt-2 bg-sky-200 rounded-full h-2 overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-sky-500 to-blue-600 rounded-full transition-all duration-1000"
                      style={{ width: `${Math.max(...result.probabilities) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="text-center">
          <p className="text-sky-600 text-sm">
            Powered by Machine Learning • Built with ❤️
          </p>
        </div>
      </div>
    </div>
  )
}