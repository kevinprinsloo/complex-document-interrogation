import { useId } from "@fluentui/react-hooks";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { TextField, ITextFieldProps, Checkbox, ICheckboxProps, Dropdown, IDropdownProps, IDropdownOption, Stack, PrimaryButton } from "@fluentui/react";
import { Search20Regular } from "@fluentui/react-icons";
import { LanguagePicker } from "../../i18n/LanguagePicker";
import { HelpCallout } from "../HelpCallout";
import { VectorSettings } from "../VectorSettings";
import { FilterModal, FilterCriteria } from "../FilterModal/FilterModal";
import { RetrievalMode } from "../../api";
import styles from "./Settings.module.css";

// Add type for onRenderLabel
type RenderLabelType = ITextFieldProps | IDropdownProps | ICheckboxProps;

export interface SettingsProps {
    promptTemplate: string;
    temperature: number;
    retrieveCount: number;
    resultsMergeStrategy: string;
    seed: number | null;
    minimumSearchScore: number;
    minimumRerankerScore: number;
    useSemanticRanker: boolean;
    useSemanticCaptions: boolean;
    useQueryRewriting: boolean;
    reasoningEffort: string;
    excludeCategory: string;
    includeCategory: string;
    includeDocumentType: string;
    includeYear: string;
    includeVendor: string;
    retrievalMode: RetrievalMode;
    sendTextSources: boolean;
    sendImageSources: boolean;
    searchTextEmbeddings: boolean;
    searchImageEmbeddings: boolean;
    showSemanticRankerOption: boolean;
    showQueryRewritingOption: boolean;
    showReasoningEffortOption: boolean;
    showMultimodalOptions: boolean;
    showVectorOption: boolean;
    useOidSecurityFilter: boolean;
    useGroupsSecurityFilter: boolean;
    useLogin: boolean;
    loggedIn: boolean;
    requireAccessControl: boolean;
    className?: string;
    onChange: (field: string, value: any) => void;
    streamingEnabled?: boolean; // Only used in chat
    shouldStream?: boolean; // Only used in Chat
    useSuggestFollowupQuestions?: boolean; // Only used in Chat
    promptTemplatePrefix?: string;
    promptTemplateSuffix?: string;
    showSuggestFollowupQuestions?: boolean;
    showAgenticRetrievalOption: boolean;
    useAgenticRetrieval: boolean;
    showLanguagePicker?: boolean;
    // New filter-related props
    advancedFilters?: FilterCriteria;
    onAdvancedFiltersChange?: (filters: FilterCriteria) => void;
}

export const Settings = ({
    promptTemplate,
    temperature,
    retrieveCount,
    resultsMergeStrategy,
    seed,
    minimumSearchScore,
    minimumRerankerScore,
    useSemanticRanker,
    useSemanticCaptions,
    useQueryRewriting,
    reasoningEffort,
    excludeCategory,
    includeCategory,
    includeDocumentType,
    includeYear,
    includeVendor,
    retrievalMode,
    searchTextEmbeddings,
    searchImageEmbeddings,
    sendTextSources,
    sendImageSources,
    showSemanticRankerOption,
    showQueryRewritingOption,
    showReasoningEffortOption,
    showMultimodalOptions,
    showVectorOption,
    useOidSecurityFilter,
    useGroupsSecurityFilter,
    useLogin,
    loggedIn,
    requireAccessControl,
    className,
    onChange,
    streamingEnabled,
    shouldStream,
    useSuggestFollowupQuestions,
    promptTemplatePrefix,
    promptTemplateSuffix,
    showSuggestFollowupQuestions,
    showAgenticRetrievalOption,
    useAgenticRetrieval,
    showLanguagePicker,
    advancedFilters,
    onAdvancedFiltersChange
}: SettingsProps) => {
    const { t, i18n } = useTranslation();
    const [isFilterModalOpen, setIsFilterModalOpen] = useState(false);
    
    // Initialize advanced filters if not provided
    const currentAdvancedFilters: FilterCriteria = advancedFilters || {
        category: includeCategory ? [includeCategory] : [],
        documenttype: includeDocumentType ? [includeDocumentType] : [],
        year: includeYear ? [includeYear] : [],
        vendor: includeVendor ? [includeVendor] : [],
        selectedFiles: [],
        selectedPrompt: ''
    };

    // Form field IDs
    const promptTemplateId = useId("promptTemplate");
    const promptTemplateFieldId = useId("promptTemplateField");
    const temperatureId = useId("temperature");
    const temperatureFieldId = useId("temperatureField");
    const seedId = useId("seed");
    const seedFieldId = useId("seedField");
    const agenticRetrievalId = useId("agenticRetrieval");
    const agenticRetrievalFieldId = useId("agenticRetrievalField");
    const searchScoreId = useId("searchScore");
    const searchScoreFieldId = useId("searchScoreField");
    const rerankerScoreId = useId("rerankerScore");
    const rerankerScoreFieldId = useId("rerankerScoreField");
    const retrieveCountId = useId("retrieveCount");
    const retrieveCountFieldId = useId("retrieveCountField");
    const resultsMergeStrategyFieldId = useId("resultsMergeStrategy");
    const includeCategoryId = useId("includeCategory");
    const includeCategoryFieldId = useId("includeCategoryField");
    const excludeCategoryId = useId("excludeCategory");
    const excludeCategoryFieldId = useId("excludeCategoryField");
    const includeDocumentTypeId = useId("includeDocumentType");
    const includeDocumentTypeFieldId = useId("includeDocumentTypeField");
    const includeYearId = useId("includeYear");
    const includeYearFieldId = useId("includeYearField");
    const includeVendorId = useId("includeVendor");
    const includeVendorFieldId = useId("includeVendorField");
    const semanticRankerId = useId("semanticRanker");
    const semanticRankerFieldId = useId("semanticRankerField");
    const queryRewritingFieldId = useId("queryRewritingField");
    const reasoningEffortFieldId = useId("reasoningEffortField");
    const semanticCaptionsId = useId("semanticCaptions");
    const semanticCaptionsFieldId = useId("semanticCaptionsField");
    const useOidSecurityFilterId = useId("useOidSecurityFilter");
    const useOidSecurityFilterFieldId = useId("useOidSecurityFilterField");
    const useGroupsSecurityFilterId = useId("useGroupsSecurityFilter");
    const useGroupsSecurityFilterFieldId = useId("useGroupsSecurityFilterField");
    const shouldStreamId = useId("shouldStream");
    const shouldStreamFieldId = useId("shouldStreamField");
    const suggestFollowupQuestionsId = useId("suggestFollowupQuestions");
    const suggestFollowupQuestionsFieldId = useId("suggestFollowupQuestionsField");

    const renderLabel = (props: RenderLabelType | undefined, labelId: string, fieldId: string, helpText: string) => (
        <HelpCallout labelId={labelId} fieldId={fieldId} helpText={helpText} label={props?.label} />
    );

    return (
        <div className={`${styles.settingsContainer} ${className}`}>
            <div className={styles.settingsSection}>
                <h3 className={styles.sectionHeader}>⚙️ {t("overallSettings")}</h3>

            {shouldStream !== undefined && (
                <Checkbox
                    id={shouldStreamFieldId}
                    disabled={!streamingEnabled}
                    className={styles.settingsSeparator}
                    checked={shouldStream}
                    label={t("labels.shouldStream")}
                    onChange={(_ev, checked) => onChange("shouldStream", !!checked)}
                    aria-labelledby={shouldStreamId}
                    onRenderLabel={props => renderLabel(props, shouldStreamId, shouldStreamFieldId, t("helpTexts.streamChat"))}
                />
            )}

            {showSuggestFollowupQuestions && (
                <Checkbox
                    id={suggestFollowupQuestionsFieldId}
                    className={styles.settingsSeparator}
                    checked={useSuggestFollowupQuestions}
                    label={t("labels.useSuggestFollowupQuestions")}
                    onChange={(_ev, checked) => onChange("useSuggestFollowupQuestions", !!checked)}
                    aria-labelledby={suggestFollowupQuestionsId}
                    onRenderLabel={props =>
                        renderLabel(props, suggestFollowupQuestionsId, suggestFollowupQuestionsFieldId, t("helpTexts.suggestFollowupQuestions"))
                    }
                />
            )}

            {showLanguagePicker && (
                <div className={styles.settingsSeparator}>
                    <h4 className={styles.subsectionHeader}>Language Settings</h4>
                    <LanguagePicker onLanguageChange={newLang => i18n.changeLanguage(newLang)} />
                </div>
            )}
            </div>

            <div className={styles.settingsSection}>
                <h3 className={styles.sectionHeader}><Search20Regular style={{ marginRight: '8px', verticalAlign: 'middle' }} />{t("searchSettings")}</h3>

            {showAgenticRetrievalOption && (
                <Checkbox
                    id={agenticRetrievalFieldId}
                    className={styles.settingsSeparator}
                    checked={useAgenticRetrieval}
                    label={t("labels.useAgenticRetrieval")}
                    onChange={(_ev, checked) => onChange("useAgenticRetrieval", !!checked)}
                    aria-labelledby={agenticRetrievalId}
                    onRenderLabel={props => renderLabel(props, agenticRetrievalId, agenticRetrievalFieldId, t("helpTexts.useAgenticRetrieval"))}
                />
            )}
            {!useAgenticRetrieval && (
                <TextField
                    id={searchScoreFieldId}
                    className={styles.settingsSeparator}
                    label={t("labels.minimumSearchScore")}
                    type="number"
                    min={0}
                    step={0.01}
                    defaultValue={minimumSearchScore.toString()}
                    onChange={(_ev, val) => onChange("minimumSearchScore", parseFloat(val || "0"))}
                    aria-labelledby={searchScoreId}
                    onRenderLabel={props => renderLabel(props, searchScoreId, searchScoreFieldId, t("helpTexts.searchScore"))}
                />
            )}

            {showSemanticRankerOption && (
                <TextField
                    id={rerankerScoreFieldId}
                    className={styles.settingsSeparator}
                    label={t("labels.minimumRerankerScore")}
                    type="number"
                    min={1}
                    max={4}
                    step={0.1}
                    defaultValue={minimumRerankerScore.toString()}
                    onChange={(_ev, val) => onChange("minimumRerankerScore", parseFloat(val || "0"))}
                    aria-labelledby={rerankerScoreId}
                    onRenderLabel={props => renderLabel(props, rerankerScoreId, rerankerScoreFieldId, t("helpTexts.rerankerScore"))}
                />
            )}
            {showAgenticRetrievalOption && useAgenticRetrieval && (
                <Dropdown
                    id={resultsMergeStrategyFieldId}
                    className={styles.settingsSeparator}
                    label={t("labels.resultsMergeStrategy")}
                    selectedKey={resultsMergeStrategy}
                    onChange={(_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, option?: IDropdownOption) =>
                        onChange("resultsMergeStrategy", option?.key)
                    }
                    aria-labelledby={includeCategoryId}
                    options={[
                        { key: "interleaved", text: t("labels.resultsMergeStrategyOptions.interleaved") },
                        { key: "descending", text: t("labels.resultsMergeStrategyOptions.descending") }
                    ]}
                    onRenderLabel={props => renderLabel(props, includeCategoryId, includeCategoryFieldId, t("helpTexts.resultsMergeStrategy"))}
                />
            )}
            <TextField
                id={retrieveCountFieldId}
                className={styles.settingsSeparator}
                label={t("labels.retrieveCount")}
                type="number"
                min={1}
                max={50}
                defaultValue={retrieveCount.toString()}
                onChange={(_ev, val) => onChange("retrieveCount", parseInt(val || "3"))}
                aria-labelledby={retrieveCountId}
                onRenderLabel={props => renderLabel(props, retrieveCountId, retrieveCountFieldId, t("helpTexts.retrieveNumber"))}
            />
            {/* Advanced Filters Section */}
            <div className={styles.settingsSeparator}>
                <h4 className={styles.subsectionHeader}>Document Filters</h4>
                <p className={styles.filterDescription}>
                    Use advanced filters to narrow down your search results by category, document type, year, vendor, or specific files.
                </p>
                
                {/* Current filter summary */}
                {(currentAdvancedFilters.category.length > 0 || 
                    currentAdvancedFilters.documenttype.length > 0 || 
                    currentAdvancedFilters.year.length > 0 || 
                    currentAdvancedFilters.vendor.length > 0 || 
                    currentAdvancedFilters.selectedFiles.length > 0 ||
                    (currentAdvancedFilters.selectedPrompt && currentAdvancedFilters.selectedPrompt !== '')) && (
                    <div className={styles.currentFiltersDisplay}>
                        <strong>Active Filters:</strong>
                        {currentAdvancedFilters.category.length > 0 && (
                            <span className={styles.filterSummary}>Categories: {currentAdvancedFilters.category.join(', ')}</span>
                        )}
                        {currentAdvancedFilters.documenttype.length > 0 && (
                            <span className={styles.filterSummary}>Types: {currentAdvancedFilters.documenttype.join(', ')}</span>
                        )}
                        {currentAdvancedFilters.year.length > 0 && (
                            <span className={styles.filterSummary}>Years: {currentAdvancedFilters.year.join(', ')}</span>
                        )}
                        {currentAdvancedFilters.vendor.length > 0 && (
                            <span className={styles.filterSummary}>Vendors: {currentAdvancedFilters.vendor.join(', ')}</span>
                        )}
                        {currentAdvancedFilters.selectedFiles.length > 0 && (
                            <span className={styles.filterSummary}>Files: {currentAdvancedFilters.selectedFiles.length} selected</span>
                        )}
                        {currentAdvancedFilters.selectedPrompt && currentAdvancedFilters.selectedPrompt !== '' && (
                            <span className={styles.filterSummary}>Custom Prompt: {currentAdvancedFilters.selectedPrompt}</span>
                        )}
                    </div>
                )}
                
                <PrimaryButton
                    text="Configure Advanced Filters"
                    onClick={() => setIsFilterModalOpen(true)}
                    iconProps={{ iconName: 'Filter' }}
                    className={styles.advancedFiltersButton}
                />
            </div>
            
            {/* Legacy single-select dropdowns - kept for backward compatibility */}
            <TextField
                id={excludeCategoryFieldId}
                className={styles.settingsSeparator}
                label={t("labels.excludeCategory")}
                defaultValue={excludeCategory}
                onChange={(_ev, val) => onChange("excludeCategory", val || "")}
                aria-labelledby={excludeCategoryId}
                onRenderLabel={props => renderLabel(props, excludeCategoryId, excludeCategoryFieldId, t("helpTexts.excludeCategory"))}
            />
            {showSemanticRankerOption && !useAgenticRetrieval && (
                <>
                    <Checkbox
                        id={semanticRankerFieldId}
                        className={styles.settingsSeparator}
                        checked={useSemanticRanker}
                        label={t("labels.useSemanticRanker")}
                        onChange={(_ev, checked) => onChange("useSemanticRanker", !!checked)}
                        aria-labelledby={semanticRankerId}
                        onRenderLabel={props => renderLabel(props, semanticRankerId, semanticRankerFieldId, t("helpTexts.useSemanticReranker"))}
                    />

                    <Checkbox
                        id={semanticCaptionsFieldId}
                        className={styles.settingsSeparator}
                        checked={useSemanticCaptions}
                        label={t("labels.useSemanticCaptions")}
                        onChange={(_ev, checked) => onChange("useSemanticCaptions", !!checked)}
                        disabled={!useSemanticRanker}
                        aria-labelledby={semanticCaptionsId}
                        onRenderLabel={props => renderLabel(props, semanticCaptionsId, semanticCaptionsFieldId, t("helpTexts.useSemanticCaptions"))}
                    />
                </>
            )}
            {showQueryRewritingOption && !useAgenticRetrieval && (
                <>
                    <Checkbox
                        id={queryRewritingFieldId}
                        className={styles.settingsSeparator}
                        checked={useQueryRewriting}
                        disabled={!useSemanticRanker}
                        label={t("labels.useQueryRewriting")}
                        onChange={(_ev, checked) => onChange("useQueryRewriting", !!checked)}
                        aria-labelledby={queryRewritingFieldId}
                        onRenderLabel={props => renderLabel(props, queryRewritingFieldId, queryRewritingFieldId, t("helpTexts.useQueryRewriting"))}
                    />
                </>
            )}
            {showReasoningEffortOption && (
                <Dropdown
                    id={reasoningEffortFieldId}
                    selectedKey={reasoningEffort}
                    label={t("labels.reasoningEffort")}
                    onChange={(_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, option?: IDropdownOption) =>
                        onChange("reasoningEffort", option?.key || "")
                    }
                    aria-labelledby={reasoningEffortFieldId}
                    options={[
                        { key: "minimal", text: t("labels.reasoningEffortOptions.minimal") },
                        { key: "low", text: t("labels.reasoningEffortOptions.low") },
                        { key: "medium", text: t("labels.reasoningEffortOptions.medium") },
                        { key: "high", text: t("labels.reasoningEffortOptions.high") }
                    ]}
                    onRenderLabel={props => renderLabel(props, queryRewritingFieldId, queryRewritingFieldId, t("helpTexts.reasoningEffort"))}
                />
            )}
            {useLogin && (
                <>
                    <Checkbox
                        id={useOidSecurityFilterFieldId}
                        className={styles.settingsSeparator}
                        checked={useOidSecurityFilter || requireAccessControl}
                        label={t("labels.useOidSecurityFilter")}
                        disabled={!loggedIn || requireAccessControl}
                        onChange={(_ev, checked) => onChange("useOidSecurityFilter", !!checked)}
                        aria-labelledby={useOidSecurityFilterId}
                        onRenderLabel={props => renderLabel(props, useOidSecurityFilterId, useOidSecurityFilterFieldId, t("helpTexts.useOidSecurityFilter"))}
                    />
                    <Checkbox
                        id={useGroupsSecurityFilterFieldId}
                        className={styles.settingsSeparator}
                        checked={useGroupsSecurityFilter || requireAccessControl}
                        label={t("labels.useGroupsSecurityFilter")}
                        disabled={!loggedIn || requireAccessControl}
                        onChange={(_ev, checked) => onChange("useGroupsSecurityFilter", !!checked)}
                        aria-labelledby={useGroupsSecurityFilterId}
                        onRenderLabel={props =>
                            renderLabel(props, useGroupsSecurityFilterId, useGroupsSecurityFilterFieldId, t("helpTexts.useGroupsSecurityFilter"))
                        }
                    />
                </>
            )}
            {showVectorOption && !useAgenticRetrieval && (
                <>
                    <VectorSettings
                        defaultRetrievalMode={retrievalMode}
                        defaultSearchTextEmbeddings={searchTextEmbeddings}
                        defaultSearchImageEmbeddings={searchImageEmbeddings}
                        showImageOptions={showMultimodalOptions}
                        updateRetrievalMode={val => onChange("retrievalMode", val)}
                        updateSearchTextEmbeddings={val => onChange("searchTextEmbeddings", val)}
                        updateSearchImageEmbeddings={val => onChange("searchImageEmbeddings", val)}
                    />
                </>
            )}

            <TextField
                id={promptTemplateFieldId}
                className={styles.settingsSeparator}
                defaultValue={promptTemplate}
                label={t("labels.promptTemplate")}
                multiline
                autoAdjustHeight
                onChange={(_ev, val) => onChange("promptTemplate", val || "")}
                aria-labelledby={promptTemplateId}
                onRenderLabel={props => renderLabel(props, promptTemplateId, promptTemplateFieldId, t("helpTexts.promptTemplate"))}
            />
            <TextField
                id={temperatureFieldId}
                className={styles.settingsSeparator}
                label={t("labels.temperature")}
                type="number"
                min={0}
                max={1}
                step={0.1}
                defaultValue={temperature.toString()}
                onChange={(_ev, val) => onChange("temperature", parseFloat(val || "0"))}
                aria-labelledby={temperatureId}
                onRenderLabel={props => renderLabel(props, temperatureId, temperatureFieldId, t("helpTexts.temperature"))}
            />
            <TextField
                id={seedFieldId}
                className={styles.settingsSeparator}
                label={t("labels.seed")}
                type="text"
                defaultValue={seed?.toString() || ""}
                onChange={(_ev, val) => onChange("seed", val ? parseInt(val) : null)}
                aria-labelledby={seedId}
                onRenderLabel={props => renderLabel(props, seedId, seedFieldId, t("helpTexts.seed"))}
            />

            {showMultimodalOptions && !useAgenticRetrieval && (
                <fieldset className={styles.fieldset + " " + styles.settingsSeparator}>
                    <legend className={styles.legend}>{t("labels.llmInputs")}</legend>
                    <Stack tokens={{ childrenGap: 8 }}>
                        <Checkbox
                            id="sendTextSources"
                            label={t("labels.llmInputsOptions.texts")}
                            checked={sendTextSources}
                            onChange={(_ev, checked) => {
                                onChange("sendTextSources", !!checked);
                            }}
                            onRenderLabel={props => renderLabel(props, "sendTextSourcesLabel", "sendTextSources", t("helpTexts.llmTextInputs"))}
                        />
                        <Checkbox
                            id="sendImageSources"
                            label={t("labels.llmInputsOptions.images")}
                            checked={sendImageSources}
                            onChange={(_ev, checked) => {
                                onChange("sendImageSources", !!checked);
                            }}
                            onRenderLabel={props => renderLabel(props, "sendImageSourcesLabel", "sendImageSources", t("helpTexts.llmImageInputs"))}
                        />
                    </Stack>
                </fieldset>
            )}
            </div>
            
            {/* Filter Modal */}
            {onAdvancedFiltersChange && (
                <FilterModal
                    isOpen={isFilterModalOpen}
                    onDismiss={() => setIsFilterModalOpen(false)}
                    selectedFilters={currentAdvancedFilters}
                    onFiltersChange={(filters) => {
                        // Update legacy single-select fields for backward compatibility
                        if (filters.category.length > 0 && filters.category[0] !== includeCategory) {
                            onChange("includeCategory", filters.category[0]);
                        } else if (filters.category.length === 0 && includeCategory) {
                            onChange("includeCategory", "");
                        }
                        
                        if (filters.documenttype.length > 0 && filters.documenttype[0] !== includeDocumentType) {
                            onChange("includeDocumentType", filters.documenttype[0]);
                        } else if (filters.documenttype.length === 0 && includeDocumentType) {
                            onChange("includeDocumentType", "");
                        }
                        
                        if (filters.year.length > 0 && filters.year[0] !== includeYear) {
                            onChange("includeYear", filters.year[0]);
                        } else if (filters.year.length === 0 && includeYear) {
                            onChange("includeYear", "");
                        }
                        
                        if (filters.vendor.length > 0 && filters.vendor[0] !== includeVendor) {
                            onChange("includeVendor", filters.vendor[0]);
                        } else if (filters.vendor.length === 0 && includeVendor) {
                            onChange("includeVendor", "");
                        }
                        
                        // Call the advanced filters change handler
                        onAdvancedFiltersChange(filters);
                    }}
                    onApplyFilters={(filters) => {
                        if (onAdvancedFiltersChange) {
                            onAdvancedFiltersChange(filters);
                        }
                    }}
                />
            )}
        </div>
    );
};
