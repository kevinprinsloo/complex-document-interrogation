import { useState, useCallback } from "react";
import { ProcessingStep } from "../components/ProcessingPanel";

export interface UseProcessingStepsReturn {
    steps: ProcessingStep[];
    currentStep: string | undefined;
    isProcessing: boolean;
    addStep: (step: ProcessingStep) => void;
    updateStep: (id: string, updates: Partial<ProcessingStep>) => void;
    clearSteps: () => void;
    setProcessing: (processing: boolean) => void;
}

export const useProcessingSteps = (): UseProcessingStepsReturn => {
    const [steps, setSteps] = useState<ProcessingStep[]>([]);
    const [currentStep, setCurrentStep] = useState<string | undefined>();
    const [isProcessing, setIsProcessing] = useState(false);

    const addStep = useCallback((step: ProcessingStep) => {
        setSteps(prev => {
            // Update current step to this new one if it's in progress
            if (step.status === "in_progress") {
                setCurrentStep(step.id);
            }
            return [...prev, step];
        });
    }, []);

    const updateStep = useCallback((id: string, updates: Partial<ProcessingStep>) => {
        setSteps(prev => prev.map(step => {
            if (step.id === id) {
                const updatedStep = { ...step, ...updates };
                // Update current step tracking
                if (updatedStep.status === "in_progress") {
                    setCurrentStep(id);
                } else if (currentStep === id && updatedStep.status === "completed") {
                    setCurrentStep(undefined);
                }
                return updatedStep;
            }
            return step;
        }));
    }, [currentStep]);

    const clearSteps = useCallback(() => {
        setSteps([]);
        setCurrentStep(undefined);
        setIsProcessing(false);
    }, []);

    const setProcessing = useCallback((processing: boolean) => {
        setIsProcessing(processing);
        if (!processing) {
            // Mark any in-progress steps as completed when processing stops
            setSteps(prev => prev.map(step => 
                step.status === "in_progress" 
                    ? { ...step, status: "completed" as const, timestamp: new Date() }
                    : step
            ));
            setCurrentStep(undefined);
        }
    }, []);

    return {
        steps,
        currentStep,
        isProcessing,
        addStep,
        updateStep,
        clearSteps,
        setProcessing
    };
};
