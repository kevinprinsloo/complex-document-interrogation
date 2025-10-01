import React, { useState, useEffect, useRef } from 'react';
import { Modal, IconButton, PrimaryButton, DefaultButton, Spinner, MessageBar, MessageBarType, Dropdown, IDropdownOption } from '@fluentui/react';
import { Settings20Regular, Flash20Regular, Folder20Regular, Tag20Regular, Document20Regular, Calendar20Regular, Building20Regular, ClipboardTask20Regular } from '@fluentui/react-icons';
import styles from './FilterModal.module.css';

// Types
interface OptionType {
  value: string;
  label: string;
}

export interface FilterCriteria {
  category: string[];
  documenttype: string[];
  year: string[];
  vendor: string[];
  selectedFiles: string[];
  selectedPrompt?: string;
}

interface AdvancedFilters {
  category: string[];
  documenttype: string[];
  year: string[];
  vendor: string[];
}

interface FilterOptionsData {
  categories: OptionType[];
  document_types: OptionType[];
  years: OptionType[];
  vendors: OptionType[];
  files: OptionType[];
  prompts: OptionType[];
}

export interface DocumentMetadata {
  id: string;
  title?: string | null;
  source_url?: string | null;
  category?: string | null;
  documenttype?: string | null;
  year?: string | null;
  vendor?: string | null;
}

export interface CustomPrompt {
  id: string;
  name: string;
  description: string;
  category: string;
  author: string;
  version: string;
}

interface FilterModalProps {
  isOpen: boolean;
  onDismiss: () => void;
  selectedFilters: FilterCriteria;
  onFiltersChange: (filters: FilterCriteria) => void;
  onApplyFilters: (filters: FilterCriteria) => void;
}


// Selected Filters Display Component
interface SelectedFiltersDisplayProps {
  filters: FilterCriteria;
  onRemove: (filterKey: keyof FilterCriteria, valueToRemove: string) => void;
  fileOptions?: OptionType[];
}

const SelectedFiltersDisplay: React.FC<SelectedFiltersDisplayProps> = ({ filters, onRemove, fileOptions = [] }) => {
  const activeFilterEntries = (Object.entries(filters) as [keyof FilterCriteria, string | string[]][])
    .filter(([key, values]) => {
      if (key === 'selectedPrompt') {
        return values && values !== '';
      }
      return Array.isArray(values) && values.length > 0;
    });

  if (activeFilterEntries.length === 0) {
    return <p className={styles.noFiltersText}>No filters currently selected.</p>;
  }

  return (
    <div className={styles.selectedFiltersContainer}>
      <h5 className={styles.selectedFiltersTitle}>Active Filters:</h5>
      {activeFilterEntries.map(([key, values]) => (
        <div key={key} className={styles.selectedFilterGroup}>
          <span className={styles.selectedFilterKey}>{key.charAt(0).toUpperCase() + key.slice(1)}:</span>
          {(() => {
            const valuesToDisplay = key === 'selectedPrompt' ? [values as string] : values as string[];
            return valuesToDisplay.map((value: string) => {
              const modifierClass =
                key === 'category' ? styles.selectedFilterValueCategory :
                key === 'documenttype' ? styles.selectedFilterValueDocumenttype :
                key === 'year' ? styles.selectedFilterValueYear :
                key === 'vendor' ? styles.selectedFilterValueVendor :
                key === 'selectedFiles' ? styles.selectedFilterValueSelectedFiles :
                key === 'selectedPrompt' ? styles.selectedFilterValueSelectedPrompt : '';
              let displayValue = value;
              if (key === 'selectedFiles' && fileOptions) {
                const fileOption = fileOptions.find(f => f.value === value);
                displayValue = fileOption ? fileOption.label : value;
              }
              return (
                <span key={value} className={`${styles.selectedFilterValue} ${modifierClass}`}>
                  {displayValue}
                  <button 
                    onClick={() => onRemove(key, value)} 
                    className={styles.removeFilterButton}
                    aria-label={`Remove filter ${displayValue}`}
                  >
                    ×
                  </button>
                </span>
              );
            });
          })()}
        </div>
      ))}
    </div>
  );
};

// Selected Files Metadata Panel
interface SelectedFilesMetadataPanelProps {
  selectedFileIds: string[];
  allDocuments: DocumentMetadata[];
}

const SelectedFilesMetadataPanel: React.FC<SelectedFilesMetadataPanelProps> = ({ selectedFileIds, allDocuments }) => {
  if (!selectedFileIds || selectedFileIds.length === 0) {
    return null;
  }

  const selectedDocuments = allDocuments.filter(doc => selectedFileIds.includes(doc.id));

  if (selectedDocuments.length === 0) {
    return <p className={`${styles.noFiltersText} mt-4 text-red-500`}>No metadata to display for selected files (files not found in metadata list).</p>;
  }

  return (
    <div className={`${styles.selectedFiltersContainer} mt-6 border-t border-gray-200 pt-4`}>
      <h5 className={`${styles.selectedFiltersTitle} mb-3`}>Selected Files Details:</h5>
      <ul className="space-y-3">
        {selectedDocuments.map(doc => (
          <li key={doc.id} className="p-3 bg-gray-100 rounded-md shadow">
            <p className="font-semibold text-gray-800 break-all">{doc.title || doc.id}</p>
            <div className="text-xs text-gray-600 mt-1 space-y-0.5">
              {doc.category && <p><strong>Category:</strong> {doc.category}</p>}
              {doc.documenttype && <p><strong>Type:</strong> {doc.documenttype}</p>}
              {doc.year && <p><strong>Year:</strong> {doc.year}</p>}
              {doc.vendor && <p><strong>Vendor:</strong> {doc.vendor}</p>}
              {!doc.category && !doc.documenttype && !doc.year && !doc.vendor && <p>No additional metadata available.</p>}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export const FilterModal: React.FC<FilterModalProps> = ({
  isOpen,
  onDismiss,
  selectedFilters,
  onFiltersChange,
  onApplyFilters
}) => {
  const [localFilters, setLocalFilters] = useState<FilterCriteria>(selectedFilters);
  const [filterOptionsData, setFilterOptionsData] = useState<FilterOptionsData | null>(null);
  const [allDocumentsMetadata, setAllDocumentsMetadata] = useState<DocumentMetadata[]>([]);
  const [customPrompts, setCustomPrompts] = useState<CustomPrompt[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const updateTimeoutRef = useRef<number | null>(null);

  const API_BASE_URL = 'http://localhost:50505';

  // Fetch filter options and document metadata
  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen]);

  // Update local filters when props change
  useEffect(() => {
    setLocalFilters(selectedFilters);
  }, [selectedFilters]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch custom prompts first
      const prompts = await fetchCustomPromptsData();
      setCustomPrompts(prompts);
      
      // Then fetch filter data and build complete options
      await fetchFilterDataWithPrompts(prompts);
    } catch (err: any) {
      console.error('Failed to fetch data:', err);
      setError(err.message || 'Failed to load data');
    }
    setLoading(false);
  };

  const fetchCustomPromptsData = async (): Promise<CustomPrompt[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/prompts/custom`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status} fetching custom prompts`);
      }
      const data: { prompts: CustomPrompt[] } = await response.json();
      return data.prompts || [];
    } catch (err: any) {
      console.error('Failed to fetch custom prompts:', err);
      return [];
    }
  };

  const fetchFilterDataWithPrompts = async (prompts: CustomPrompt[]) => {
    try {
      const response = await fetch(`${API_BASE_URL}/metadata/all-documents-metadata`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status} fetching document metadata`);
      }
      const data: DocumentMetadata[] = await response.json();
      
      // Remove duplicates based on document ID
      const uniqueDocuments = data.filter((doc, index, self) => 
        index === self.findIndex(d => d.id === doc.id)
      );

      const categories = Array.from(new Set(uniqueDocuments.map(doc => doc.category).filter(Boolean) as string[]))
        .sort()
        .map(val => ({ value: val, label: val }));
      
      const documentTypes = Array.from(new Set(uniqueDocuments.map(doc => doc.documenttype).filter(Boolean) as string[]))
        .sort()
        .map(val => ({ value: val, label: val }));
      
      const years = Array.from(new Set(uniqueDocuments.map(doc => doc.year).filter(Boolean) as string[]))
        .sort()
        .map(val => ({ value: val, label: val }));
      
      const vendors = Array.from(new Set(uniqueDocuments.map(doc => doc.vendor).filter(Boolean) as string[]))
        .sort()
        .map(val => ({ value: val, label: val }));
      
      const files = uniqueDocuments.map(doc => {
        let label = doc.title || doc.id;
        const metadataParts: string[] = [];
        if (doc.category) metadataParts.push(`Category: ${doc.category}`);
        if (doc.documenttype) metadataParts.push(`Type: ${doc.documenttype}`);
        if (doc.year) metadataParts.push(`Year: ${doc.year}`);
        if (doc.vendor) metadataParts.push(`Vendor: ${doc.vendor}`);
        if (metadataParts.length > 0) {
          label += ` (${metadataParts.join(', ')})`;
        }
        return { value: doc.id, label };
      }).sort((a, b) => a.label.localeCompare(b.label));

      setFilterOptionsData({
        categories,
        document_types: documentTypes,
        years,
        vendors,
        files,
        prompts: prompts.map(prompt => ({ value: prompt.id, label: `${prompt.name} (${prompt.category})` })),
      });
    } catch (err: any) {
      console.error('Failed to fetch filter data:', err);
      throw err;
    }
  };

  // Fetch filtered options based on current selections
  const fetchFilteredOptions = async (currentFilters: AdvancedFilters) => {
    try {
      const response = await fetch(`${API_BASE_URL}/metadata/filtered-options`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filters: {
            category: currentFilters.category,
            documenttype: currentFilters.documenttype,
            year: currentFilters.year,
            vendor: currentFilters.vendor
          }
        })
      });

      if (response.ok) {
        const filteredOptions = await response.json();
        
        // Get prompts separately since they don't depend on document filters
        const promptsResponse = await fetch(`${API_BASE_URL}/prompts/custom`);
        const promptsData = promptsResponse.ok ? await promptsResponse.json() : [];
        
        // Ensure prompts is always an array
        const prompts = Array.isArray(promptsData) ? promptsData : [];

        setFilterOptionsData({
          categories: filteredOptions.categories || [],
          document_types: filteredOptions.document_types || [],
          years: filteredOptions.years || [],
          vendors: filteredOptions.vendors || [],
          files: filteredOptions.files || [],
          prompts: prompts.map((prompt: CustomPrompt) => ({ value: prompt.id, label: `${prompt.name} (${prompt.category})` })),
        });
      }
    } catch (err: any) {
      console.error('Failed to fetch filtered options:', err);
      // Fall back to original method if filtered options fail
      await fetchData();
    }
  };

  // Legacy function - kept for backward compatibility but now just calls the new fetchData
  const fetchFilterData = async () => {
    await fetchData();
  };

  // Legacy function - kept for backward compatibility but now just calls the new fetchCustomPromptsData
  const fetchCustomPrompts = async () => {
    try {
      const prompts = await fetchCustomPromptsData();
      setCustomPrompts(prompts);
    } catch (err: any) {
      console.error('Failed to fetch custom prompts:', err);
      // Don't set error state for prompts failure, just log it
    }
  };

  const handleFilterChange = async (selectedOptions: OptionType[], filterName: keyof FilterCriteria) => {
    const values = selectedOptions ? selectedOptions.map(option => option.value) : [];
    const newFilters = { ...localFilters, [filterName]: values };
    setLocalFilters(newFilters);
    onFiltersChange(newFilters);
    
    // Debounce the filter options update to prevent UI flickering
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    
    updateTimeoutRef.current = setTimeout(async () => {
      await updateFilterOptions(newFilters);
    }, 2000); // Wait 2 seconds before updating options
  };

  const updateFilterOptions = async (currentFilters: FilterCriteria) => {
    try {
      // Convert FilterCriteria to AdvancedFilters format
      const advancedFilters: AdvancedFilters = {
        category: currentFilters.category,
        documenttype: currentFilters.documenttype,
        year: currentFilters.year,
        vendor: currentFilters.vendor
      };
      
      await fetchFilteredOptions(advancedFilters);
    } catch (err: any) {
      console.error('Failed to update filter options:', err);
      // Fall back to showing all options if filtering fails
    }
  };

  const handleRemoveFilter = async (filterKey: keyof FilterCriteria, valueToRemove: string) => {
    const currentValue = localFilters[filterKey];
    let newValue: string | string[];
    
    if (filterKey === 'selectedPrompt') {
      // For selectedPrompt, it's a single string, so we clear it
      newValue = '';
    } else {
      // For array filters, filter out the value
      newValue = Array.isArray(currentValue) 
        ? currentValue.filter((val: string) => val !== valueToRemove)
        : [];
    }
    
    const newFilters = {
      ...localFilters,
      [filterKey]: newValue,
    };
    setLocalFilters(newFilters);
    onFiltersChange(newFilters);
    
    // Debounce the filter options update
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    
    updateTimeoutRef.current = setTimeout(async () => {
      await updateFilterOptions(newFilters);
    }, 2000);
  };

  const handleApply = () => {
    onApplyFilters(localFilters);
    onDismiss();
  };

  const handleClear = async () => {
    const clearedFilters: FilterCriteria = {
      category: [],
      documenttype: [],
      year: [],
      vendor: [],
      selectedFiles: [],
      selectedPrompt: '',
    };
    setLocalFilters(clearedFilters);
    onFiltersChange(clearedFilters);
    
    // Clear any pending updates
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    
    // Reset to show all options when cleared
    await fetchData();
  };

  const getSelectedOptions = (allOptions: OptionType[] | undefined, selectedValues: string[]): OptionType[] => {
    if (!allOptions) return [];
    return allOptions.filter(option => selectedValues.includes(option.value));
  };

  // Ensure selected values are always available in dropdown options
  const ensureSelectedOptionsAvailable = (options: OptionType[], selectedValues: string[]): OptionType[] => {
    const optionValues = new Set(options.map(opt => opt.value));
    const missingOptions = selectedValues
      .filter(value => !optionValues.has(value))
      .map(value => ({ value, label: value }));
    
    return [...options, ...missingOptions];
  };

  return (
    <Modal
      isOpen={isOpen}
      onDismiss={onDismiss}
      isBlocking={false}
      containerClassName={styles.modalContainer}
    >
      <div className={styles.modalContent}>
        <div className={styles.modalHeader}>
          <div className={styles.modalTitle}>
            <span className={styles.filterIcon}>⚙️</span>
            <h2>Developer Settings</h2>
          </div>
          <IconButton
            iconProps={{ iconName: 'Cancel' }}
            ariaLabel="Close"
            onClick={onDismiss}
            className={styles.closeButton}
          />
        </div>

        <div className={styles.modalBody}>
          {loading && (
            <div className={styles.loadingContainer}>
              <Spinner label="Loading filter options..." />
            </div>
          )}

          {error && (
            <MessageBar messageBarType={MessageBarType.error} className={styles.errorMessage}>
              Error loading filters: {error}
            </MessageBar>
          )}

          {!loading && !error && filterOptionsData && (
            <>
              <div className={styles.filterSection}>
                <h3><Settings20Regular style={{ marginRight: '8px', verticalAlign: 'middle' }} />Prompt Selection</h3>
                
                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}><Flash20Regular style={{ marginRight: '6px', verticalAlign: 'middle' }} />Custom Prompt</label>
                  <Dropdown
                    options={[
                      { key: '', text: 'Default (System Prompt)' },
                      ...filterOptionsData.prompts.map(prompt => ({ key: prompt.value, text: prompt.label }))
                    ]}
                    selectedKey={localFilters.selectedPrompt || ''}
                    onChange={(event, option) => {
                      if (option) {
                        const newFilters = { ...localFilters, selectedPrompt: option.key as string };
                        setLocalFilters(newFilters);
                        onFiltersChange(newFilters);
                      }
                    }}
                    placeholder="Select a custom prompt..."
                    className={styles.singleSelectContainer}
                  />
                  {localFilters.selectedPrompt && (
                    <div className={styles.promptDescription}>
                      {(() => {
                        const selectedPrompt = customPrompts.find(p => p.id === localFilters.selectedPrompt);
                        return selectedPrompt ? (
                          <small className={styles.promptDescriptionText}>
                            <strong>{selectedPrompt.name}</strong>: {selectedPrompt.description}
                            <br />
                            <em>Category: {selectedPrompt.category} | Author: {selectedPrompt.author}</em>
                          </small>
                        ) : null;
                      })()}
                    </div>
                  )}
                </div>
              </div>

              <div className={styles.filterSection}>
                <h3><Folder20Regular style={{ marginRight: '8px', verticalAlign: 'middle' }} />Document Filters</h3>
                
                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}><Tag20Regular style={{ marginRight: '6px', verticalAlign: 'middle' }} />Category</label>
                  <Dropdown
                    multiSelect
                    options={ensureSelectedOptionsAvailable(filterOptionsData.categories, localFilters.category).map(cat => ({ key: cat.value, text: cat.label }))}
                    selectedKeys={localFilters.category}
                    onChange={(event, option) => {
                      if (option) {
                        const newValues = option.selected 
                          ? [...localFilters.category, option.key as string]
                          : localFilters.category.filter(v => v !== option.key);
                        handleFilterChange(newValues.map(v => ({ value: v, label: v })), 'category');
                      }
                    }}
                    placeholder="Select categories..."
                    className={styles.multiSelectContainer}
                  />
                </div>

                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}><Document20Regular style={{ marginRight: '6px', verticalAlign: 'middle' }} />Document Type</label>
                  <Dropdown
                    multiSelect
                    options={ensureSelectedOptionsAvailable(filterOptionsData.document_types, localFilters.documenttype).map(type => ({ key: type.value, text: type.label }))}
                    selectedKeys={localFilters.documenttype}
                    onChange={(event, option) => {
                      if (option) {
                        const newValues = option.selected 
                          ? [...localFilters.documenttype, option.key as string]
                          : localFilters.documenttype.filter(v => v !== option.key);
                        handleFilterChange(newValues.map(v => ({ value: v, label: v })), 'documenttype');
                      }
                    }}
                    placeholder="Select document types..."
                    className={styles.multiSelectContainer}
                  />
                </div>

                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}><Calendar20Regular style={{ marginRight: '6px', verticalAlign: 'middle' }} />Year</label>
                  <Dropdown
                    multiSelect
                    options={ensureSelectedOptionsAvailable(filterOptionsData.years, localFilters.year).map(year => ({ key: year.value, text: year.label }))}
                    selectedKeys={localFilters.year}
                    onChange={(event, option) => {
                      if (option) {
                        const newValues = option.selected 
                          ? [...localFilters.year, option.key as string]
                          : localFilters.year.filter(v => v !== option.key);
                        handleFilterChange(newValues.map(v => ({ value: v, label: v })), 'year');
                      }
                    }}
                    placeholder="Select years..."
                    className={styles.multiSelectContainer}
                  />
                </div>

                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}><Building20Regular style={{ marginRight: '6px', verticalAlign: 'middle' }} />Vendor</label>
                  <Dropdown
                    multiSelect
                    options={ensureSelectedOptionsAvailable(filterOptionsData.vendors, localFilters.vendor).map(vendor => ({ key: vendor.value, text: vendor.label }))}
                    selectedKeys={localFilters.vendor}
                    onChange={(event, option) => {
                      if (option) {
                        const newValues = option.selected 
                          ? [...localFilters.vendor, option.key as string]
                          : localFilters.vendor.filter(v => v !== option.key);
                        handleFilterChange(newValues.map(v => ({ value: v, label: v })), 'vendor');
                      }
                    }}
                    placeholder="Select vendors..."
                    className={styles.multiSelectContainer}
                  />
                </div>

                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}><ClipboardTask20Regular style={{ marginRight: '6px', verticalAlign: 'middle' }} />Specific Files</label>
                  <Dropdown
                    multiSelect
                    options={ensureSelectedOptionsAvailable(filterOptionsData.files, localFilters.selectedFiles).map(file => ({ key: file.value, text: file.label }))}
                    selectedKeys={localFilters.selectedFiles}
                    onChange={(event, option) => {
                      if (option) {
                        const newValues = option.selected 
                          ? [...localFilters.selectedFiles, option.key as string]
                          : localFilters.selectedFiles.filter(v => v !== option.key);
                        handleFilterChange(newValues.map(v => ({ value: v, label: v })), 'selectedFiles');
                      }
                    }}
                    placeholder="Select specific files..."
                    className={styles.multiSelectContainer}
                  />
                </div>
              </div>

              <SelectedFiltersDisplay 
                filters={localFilters} 
                onRemove={handleRemoveFilter} 
                fileOptions={filterOptionsData.files} 
              />

              {localFilters.selectedFiles && localFilters.selectedFiles.length > 0 && (
                <SelectedFilesMetadataPanel 
                  selectedFileIds={localFilters.selectedFiles}
                  allDocuments={allDocumentsMetadata}
                />
              )}
            </>
          )}
        </div>

        <div className={styles.modalFooter}>
          <DefaultButton onClick={handleClear} text="Clear All" />
          <div className={styles.footerButtons}>
            <DefaultButton onClick={onDismiss} text="Cancel" />
            <PrimaryButton onClick={handleApply} text="Apply Filters" />
          </div>
        </div>
      </div>
    </Modal>
  );
};
