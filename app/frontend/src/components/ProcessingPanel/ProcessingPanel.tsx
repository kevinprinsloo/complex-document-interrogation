import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { 
    IconButton, 
    Text, 
    Spinner, 
    SpinnerSize,
    Stack,
    Icon,
    Separator
} from "@fluentui/react";
import styles from "./ProcessingPanel.module.css";

export interface ProcessingStep {
    id: string;
    title: string;
    status: "pending" | "in_progress" | "completed" | "error";
    details?: string;
    timestamp?: Date;
    metadata?: any;
}

interface ProcessingPanelProps {
    isOpen: boolean;
    isProcessing: boolean;
    steps: ProcessingStep[];
    onDismiss: () => void;
    currentStep?: string;
}

export const ProcessingPanel: React.FC<ProcessingPanelProps> = ({
    isOpen,
    isProcessing,
    steps,
    onDismiss,
    currentStep
}) => {
    const { t } = useTranslation();
    const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

    const toggleStepExpansion = (stepId: string) => {
        const newExpanded = new Set(expandedSteps);
        if (newExpanded.has(stepId)) {
            newExpanded.delete(stepId);
        } else {
            newExpanded.add(stepId);
        }
        setExpandedSteps(newExpanded);
    };

    const getStepIcon = (status: ProcessingStep["status"]) => {
        switch (status) {
            case "completed":
                return <Icon iconName="CheckMark" className={styles.completedIcon} />;
            case "in_progress":
                return <Spinner size={SpinnerSize.xSmall} className={styles.progressIcon} />;
            case "error":
                return <Icon iconName="ErrorBadge" className={styles.errorIcon} />;
            case "pending":
            default:
                return <Icon iconName="Clock" className={styles.pendingIcon} />;
        }
    };

    const formatTimestamp = (timestamp?: Date) => {
        if (!timestamp) return "";
        return timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
        });
    };

    // Auto-expand current step
    useEffect(() => {
        if (currentStep && !expandedSteps.has(currentStep)) {
            setExpandedSteps(prev => new Set([...prev, currentStep]));
        }
    }, [currentStep, expandedSteps]);

    if (!isOpen) return null;

    return (
        <div className={styles.processingPanel}>
            <div className={styles.panelContent}>
                <div className={styles.header}>
                    <Stack horizontal verticalAlign="center" tokens={{ childrenGap: 8 }}>
                        {isProcessing && <Spinner size={SpinnerSize.small} />}
                        <Text variant="medium" className={styles.headerText}>
                            {isProcessing 
                                ? t("processingPanel.processing", "Processing your request...")
                                : t("processingPanel.completed", "Processing completed")
                            }
                        </Text>
                        <IconButton
                            iconProps={{ iconName: "Cancel" }}
                            onClick={onDismiss}
                            ariaLabel="Close processing panel"
                            styles={{
                                root: { marginLeft: "auto" },
                                icon: { color: "var(--colorNeutralForeground2)" }
                            }}
                        />
                    </Stack>
                </div>

                <Separator />

                <div className={styles.stepsContainer}>
                    {steps.length === 0 ? (
                        <div className={styles.emptyState}>
                            <Icon iconName="ProcessingRun" className={styles.emptyIcon} />
                            <Text variant="medium" className={styles.emptyText}>
                                {t("processingPanel.noSteps", "No processing steps available")}
                            </Text>
                        </div>
                    ) : (
                        steps.map((step, index) => (
                            <div key={step.id} className={styles.stepContainer}>
                                <div 
                                    className={`${styles.stepHeader} ${
                                        step.status === "in_progress" ? styles.activeStep : ""
                                    }`}
                                    onClick={() => toggleStepExpansion(step.id)}
                                >
                                    <Stack horizontal verticalAlign="center" tokens={{ childrenGap: 12 }}>
                                        <div className={styles.stepNumber}>
                                            {index + 1}
                                        </div>
                                        {getStepIcon(step.status)}
                                        <div className={styles.stepInfo}>
                                            <Text variant="medium" className={styles.stepTitle}>
                                                {step.title}
                                            </Text>
                                            {step.timestamp && (
                                                <Text variant="small" className={styles.stepTimestamp}>
                                                    {formatTimestamp(step.timestamp)}
                                                </Text>
                                            )}
                                        </div>
                                        <IconButton
                                            iconProps={{ 
                                                iconName: expandedSteps.has(step.id) 
                                                    ? "ChevronUp" 
                                                    : "ChevronDown" 
                                            }}
                                            className={styles.expandButton}
                                        />
                                    </Stack>
                                </div>

                                {expandedSteps.has(step.id) && (
                                    <div className={styles.stepDetails}>
                                        {step.details && (
                                            <div className={styles.detailsSection}>
                                                <Text variant="small" className={styles.detailsLabel}>
                                                    {t("processingPanel.details", "Details:")}
                                                </Text>
                                                <pre className={styles.detailsContent}>
                                                    {step.details}
                                                </pre>
                                            </div>
                                        )}
                                        
                                        {step.metadata && (
                                            <div className={styles.metadataSection}>
                                                <Text variant="small" className={styles.detailsLabel}>
                                                    {t("processingPanel.metadata", "Metadata:")}
                                                </Text>
                                                <pre className={styles.metadataContent}>
                                                    {JSON.stringify(step.metadata, null, 2)}
                                                </pre>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};
