import { useState, useEffect, useContext, useRef } from "react";
import { Stack, TextField, ITextField } from "@fluentui/react";
import { Button, Tooltip } from "@fluentui/react-components";
import { Send28Filled, Settings28Filled } from "@fluentui/react-icons";
import { useTranslation } from "react-i18next";

import styles from "./QuestionInput.module.css";
import { SpeechInput } from "./SpeechInput";
import { LoginContext } from "../../loginContext";
import { requireLogin } from "../../authConfig";

interface Props {
    onSend: (question: string) => void;
    disabled: boolean;
    initQuestion?: string;
    placeholder?: string;
    clearOnSend?: boolean;
    showSpeechInput?: boolean;
    onSettingsClick?: () => void;
}

export const QuestionInput = ({ onSend, disabled, placeholder, clearOnSend, initQuestion, showSpeechInput, onSettingsClick }: Props) => {
    const [question, setQuestion] = useState<string>("");
    const { loggedIn } = useContext(LoginContext);
    const { t } = useTranslation();
    const [isComposing, setIsComposing] = useState(false);
    const textAreaRef = useRef<ITextField>(null);

    useEffect(() => {
        initQuestion && setQuestion(initQuestion);
    }, [initQuestion]);

    // Initial resize effect
    useEffect(() => {
        if (textAreaRef.current) {
            const textArea = (textAreaRef.current as any)._textElement?.current;
            if (textArea) {
                textArea.style.height = '60px'; // Set initial minimum height
                textArea.style.overflowY = 'hidden';
            }
        }
    }, []);

    const sendQuestion = () => {
        if (disabled || !question.trim()) {
            return;
        }

        onSend(question);

        if (clearOnSend) {
            setQuestion("");
        }
    };

    const onEnterPress = (ev: React.KeyboardEvent<Element>) => {
        if (isComposing) return;

        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            sendQuestion();
        }
    };

    const handleCompositionStart = () => {
        setIsComposing(true);
    };
    const handleCompositionEnd = () => {
        setIsComposing(false);
    };

    const onQuestionChange = (_ev: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string) => {
        if (!newValue) {
            setQuestion("");
        } else if (newValue.length <= 1000) {
            setQuestion(newValue);
        }
        
        // Auto-resize textarea
        setTimeout(() => {
            if (textAreaRef.current) {
                const textArea = (textAreaRef.current as any)._textElement?.current;
                if (textArea) {
                    const maxHeight = 200; // Maximum height in pixels (about 8 lines)
                    const minHeight = 60; // Minimum height in pixels (about 2.5 lines)
                    
                    // If empty, set to minimum height
                    if (!newValue || newValue.trim() === "") {
                        textArea.style.height = minHeight + 'px';
                        textArea.style.overflowY = 'hidden';
                    } else {
                        textArea.style.height = 'auto';
                        const scrollHeight = textArea.scrollHeight;
                        
                        if (scrollHeight <= maxHeight) {
                            textArea.style.height = Math.max(scrollHeight, minHeight) + 'px';
                            textArea.style.overflowY = 'hidden';
                        } else {
                            textArea.style.height = maxHeight + 'px';
                            textArea.style.overflowY = 'auto';
                        }
                    }
                }
            }
        }, 0);
    };

    const disableRequiredAccessControl = requireLogin && !loggedIn;
    const sendQuestionDisabled = disabled || !question.trim() || disableRequiredAccessControl;

    if (disableRequiredAccessControl) {
        placeholder = "Please login to continue...";
    }

    return (
        <Stack horizontal className={styles.questionInputContainer}>
            <TextField
                className={styles.questionInputTextArea}
                disabled={disableRequiredAccessControl}
                placeholder={placeholder}
                multiline
                resizable={false}
                borderless
                value={question}
                onChange={onQuestionChange}
                onKeyDown={onEnterPress}
                onCompositionStart={handleCompositionStart}
                onCompositionEnd={handleCompositionEnd}
                componentRef={textAreaRef}
            />
            <div className={styles.questionInputButtonsContainer}>
                {onSettingsClick && (
                    <Tooltip content="Settings" relationship="label">
                        <Button 
                            size="large" 
                            icon={<Settings28Filled primaryFill="rgba(115, 118, 225, 1)" />} 
                            onClick={onSettingsClick}
                            className={styles.settingsButton}
                        />
                    </Tooltip>
                )}
                <Tooltip content={t("tooltips.submitQuestion")} relationship="label">
                    <Button size="large" icon={<Send28Filled primaryFill="rgba(115, 118, 225, 1)" />} disabled={sendQuestionDisabled} onClick={sendQuestion} />
                </Tooltip>
            </div>
            {showSpeechInput && <SpeechInput updateQuestion={setQuestion} />}
        </Stack>
    );
};
