document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    
    // Result panels
    const placeholder = document.getElementById('result-placeholder');
    const loading = document.getElementById('result-loading');
    const display = document.getElementById('result-display');
    
    // Result details
    const predClassLabel = document.getElementById('prediction-class-label');
    const confidenceText = document.getElementById('confidence-percentage');
    const gaugeFill = document.getElementById('gauge-fill-ring');
    const descBox = document.getElementById('result-desc-box');
    const explanationText = document.getElementById('result-explanation');
    const resetBtn = document.getElementById('reset-btn');
    
    // Collapsible insights
    const insightsToggle = document.getElementById('insights-toggle');
    const insightsContent = document.getElementById('insights-content');
    const insightsChevron = document.getElementById('insights-chevron');
    
    // Drawer elements
    const historyToggleBtn = document.getElementById('history-toggle-btn');
    const historyDrawer = document.getElementById('history-drawer');
    const closeDrawerBtn = document.getElementById('close-drawer-btn');
    const drawerOverlay = document.getElementById('drawer-overlay');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const historyTableBody = document.getElementById('history-table-body');

    // LocalStorage key
    const LOCAL_STORAGE_KEY = 'car_prediction_history';

    // SVG dasharray parameter
    const SVG_CIRCUMFERENCE = 314; 

    // Initial state
    loadHistory();
    fetchMetadata();

    // Cache-bust dashboard plots to force reload and bypass browser cache
    const plots = document.querySelectorAll('.responsive-plot');
    plots.forEach(img => {
        const src = img.getAttribute('src');
        if (src) {
            img.setAttribute('src', `${src.split('?')[0]}?t=${new Date().getTime()}`);
        }
    });

    // 1. Prediction Form Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Gather values
        const inputs = {
            buying: document.getElementById('buying').value,
            maint: document.getElementById('maint').value,
            doors: document.getElementById('doors').value,
            persons: document.getElementById('persons').value,
            lug_boot: document.getElementById('lug_boot').value,
            safety: document.getElementById('safety').value
        };

        // Show loading state
        placeholder.classList.add('hidden');
        display.classList.add('hidden');
        loading.classList.remove('hidden');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(inputs)
            });

            if (!response.ok) {
                let errorMessage = `Server error: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    if (errorData && errorData.message) {
                        errorMessage = errorData.message;
                    } else if (errorData && errorData.detail) {
                        errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
                    }
                } catch (e) {
                    try {
                        const errorText = await response.text();
                        if (errorText) errorMessage = errorText;
                    } catch (textErr) {}
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();

            // Successfully received prediction
            renderResult(data);
            saveToHistory(data);
        } catch (error) {
            alert(`Error running prediction: ${error.message}`);
            // Revert state
            loading.classList.add('hidden');
            placeholder.classList.remove('hidden');
        } finally {
            submitBtn.disabled = false;
        }
    });

    // 2. Render Results Flow
    function renderResult(data) {
        loading.classList.add('hidden');
        display.classList.remove('hidden');

        const prediction = data.prediction; // e.g. 'acc', 'unacc', 'good', 'vgood'
        const label = data.prediction_label; // e.g. 'Acceptable'
        const confidence = data.confidence;

        // Reset class colors on label
        predClassLabel.className = '';
        predClassLabel.textContent = label;
        predClassLabel.classList.add(`outcome-${prediction}`);

        // Reset class colors on border box
        descBox.className = 'result-description-box';
        descBox.classList.add(`outcome-${prediction}`);

        // Update gauge text and circle animation
        confidenceText.textContent = `${confidence}%`;
        
        // Gauge fill color based on outcome
        const outcomeColors = {
            unacc: '#ef4444',
            acc: '#f59e0b',
            good: '#06b6d4',
            vgood: '#10b981'
        };
        gaugeFill.style.stroke = outcomeColors[prediction] || '#3b82f6';
        
        // Calculate offset (SVG radius = 50, circumference = 2 * pi * r = 314)
        const offset = SVG_CIRCUMFERENCE - (SVG_CIRCUMFERENCE * confidence / 100);
        gaugeFill.style.strokeDashoffset = offset;

        // Handle backward compatibility for string-based explanations in history
        let explanation = data.explanation;
        if (typeof explanation === 'string') {
            explanation = {
                summary: explanation,
                top_positive_features: [],
                top_negative_features: [],
                decision_strength: 'Moderate',
                confidence_reason: 'This is a historical record evaluated before model-driven explanations were introduced.'
            };
        }

        // Display summary text
        explanationText.textContent = explanation.summary || "Vehicular parameters processed successfully.";

        // Update Strength and Reason badges
        const strengthBadge = document.getElementById('decision-strength-badge');
        const reasonBadge = document.getElementById('confidence-reason-badge');
        
        if (strengthBadge) {
            strengthBadge.textContent = `${explanation.decision_strength} Decision`;
            strengthBadge.className = `badge-strength strength-${explanation.decision_strength.toLowerCase()}`;
        }
        
        if (reasonBadge) {
            reasonBadge.textContent = explanation.confidence_reason;
        }

        // Render SHAP Lists
        const positiveList = document.getElementById('shap-positive-list');
        const negativeList = document.getElementById('shap-negative-list');
        const shapDetails = document.getElementById('shap-details');

        if (positiveList && negativeList) {
            positiveList.innerHTML = '';
            negativeList.innerHTML = '';

            const posFeatures = explanation.top_positive_features || [];
            const negFeatures = explanation.top_negative_features || [];

            // Hide section if there is no SHAP details (e.g. legacy history)
            if (posFeatures.length === 0 && negFeatures.length === 0) {
                if (shapDetails) shapDetails.classList.add('hidden');
            } else {
                if (shapDetails) shapDetails.classList.remove('hidden');

                // Render positive factors
                if (posFeatures.length === 0) {
                    positiveList.innerHTML = '<div class="shap-empty-msg">No positive influence factors contributing to this prediction.</div>';
                } else {
                    posFeatures.forEach(feat => {
                        const item = document.createElement('div');
                        item.className = 'shap-item';
                        // Math.min/max boundary checks for progress bar widths (raw score * 100 for percentage width)
                        const barWidth = Math.min(100, Math.max(1, Math.abs(feat.influence_score) * 100));
                        item.innerHTML = `
                            <div class="shap-meta">
                                <span class="shap-label">${feat.display_name} <span class="shap-feature-val">(${feat.value})</span></span>
                                <span class="shap-score positive-score">+${feat.influence_score.toFixed(4)} Impact Score</span>
                            </div>
                            <div class="shap-bar-bg">
                                <div class="shap-bar-fill positive-fill" style="width: ${barWidth}%"></div>
                            </div>
                        `;
                        positiveList.appendChild(item);
                    });
                }

                // Render negative factors
                if (negFeatures.length === 0) {
                    negativeList.innerHTML = '<div class="shap-empty-msg">No negative influence factors opposing this prediction.</div>';
                } else {
                    negFeatures.forEach(feat => {
                        const item = document.createElement('div');
                        item.className = 'shap-item';
                        const barWidth = Math.min(100, Math.max(1, Math.abs(feat.influence_score) * 100));
                        item.innerHTML = `
                            <div class="shap-meta">
                                <span class="shap-label">${feat.display_name} <span class="shap-feature-val">(${feat.value})</span></span>
                                <span class="shap-score negative-score">${feat.influence_score.toFixed(4)} Impact Score</span>
                            </div>
                            <div class="shap-bar-bg">
                                <div class="shap-bar-fill negative-fill" style="width: ${barWidth}%"></div>
                            </div>
                        `;
                        negativeList.appendChild(item);
                    });
                }
            }
        }
    }

    // Reset Result View
    resetBtn.addEventListener('click', () => {
        display.classList.add('hidden');
        placeholder.classList.remove('hidden');
        form.reset();
    });

    // 3. Collapsible insights handler
    insightsToggle.addEventListener('click', () => {
        insightsContent.classList.toggle('hidden');
        insightsChevron.classList.toggle('rotate');
    });

    // 4. Drawer handlers (History)
    historyToggleBtn.addEventListener('click', toggleDrawer);
    closeDrawerBtn.addEventListener('click', toggleDrawer);
    drawerOverlay.addEventListener('click', toggleDrawer);

    function toggleDrawer() {
        historyDrawer.classList.toggle('hidden-drawer');
        drawerOverlay.classList.toggle('hidden');
    }

    // Clear History Logic
    clearHistoryBtn.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear all history records?")) {
            localStorage.removeItem(LOCAL_STORAGE_KEY);
            loadHistory();
        }
    });

    // Load History from localStorage
    function loadHistory() {
        const records = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY)) || [];
        historyTableBody.innerHTML = '';

        if (records.length === 0) {
            historyTableBody.innerHTML = `
                <tr class="empty-history-row">
                    <td colspan="4">No evaluations recorded yet.</td>
                </tr>
            `;
            return;
        }

        // Display newest records first
        records.reverse().forEach((record, index) => {
            const tr = document.createElement('tr');
            
            // Format parameters
            const paramList = `
                <div class="history-params">
                    <span><strong>Buy:</strong> ${record.inputs.buying.toUpperCase()} | <strong>Maint:</strong> ${record.inputs.maint.toUpperCase()}</span>
                    <span><strong>Safety:</strong> ${record.inputs.safety.toUpperCase()} | <strong>Cap:</strong> ${record.inputs.persons} pax</span>
                </div>
            `;

            // Result badge
            const outcomeColors = {
                unacc: 'outcome-unacc',
                acc: 'outcome-acc',
                good: 'outcome-good',
                vgood: 'outcome-vgood'
            };
            const colorClass = outcomeColors[record.prediction] || '';
            const badge = `<span class="badge-outcome ${colorClass}">${record.prediction.toUpperCase()}</span>`;

            tr.innerHTML = `
                <td>${paramList}</td>
                <td>${badge}</td>
                <td><strong>${record.confidence}%</strong></td>
                <td>
                    <button class="btn-icon-only btn-reapply" data-index="${records.length - 1 - index}">
                        <i class="fa-solid fa-redo"></i>
                    </button>
                </td>
            `;

            historyTableBody.appendChild(tr);
        });

        // Add event listeners to reapplying buttons
        document.querySelectorAll('.btn-reapply').forEach(button => {
            button.addEventListener('click', (e) => {
                const idx = e.currentTarget.getAttribute('data-index');
                reapplyRecord(idx);
            });
        });
    }

    // Save Prediction to localStorage
    function saveToHistory(record) {
        const records = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY)) || [];
        
        // Limit history to 20 items
        if (records.length >= 20) {
            records.shift();
        }
        
        records.push(record);
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(records));
        loadHistory();
    }

    // Reapply Past Prediction to Form
    function reapplyRecord(index) {
        const records = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY)) || [];
        const record = records[index];
        if (!record) return;

        // Set form options
        document.getElementById('buying').value = record.inputs.buying;
        document.getElementById('maint').value = record.inputs.maint;
        document.getElementById('doors').value = record.inputs.doors;
        document.getElementById('persons').value = record.inputs.persons;
        document.getElementById('lug_boot').value = record.inputs.lug_boot;
        document.getElementById('safety').value = record.inputs.safety;

        // Render result instantly
        placeholder.classList.add('hidden');
        loading.classList.add('hidden');
        renderResult(record);

        // Close drawer
        toggleDrawer();
    }

    // 5. Fetch and render model metadata dynamically
    async function fetchMetadata() {
        try {
            const response = await fetch('/api/metadata');
            if (!response.ok) {
                let errorMessage = `Server error: ${response.status} ${response.statusText}`;
                try {
                    const errorText = await response.text();
                    if (errorText) errorMessage = errorText;
                } catch (e) {}
                throw new Error(errorMessage);
            }
            const data = await response.json();
            if (data.status === 'success') {
                const meta = data.metadata;
                
                // Update metrics indicators in insights panel
                const accVal = document.getElementById('accuracy-value');
                const sizeVal = document.getElementById('size-value');
                if (accVal) accVal.textContent = `${(meta.accuracy * 100).toFixed(2)}%`;
                if (sizeVal) sizeVal.textContent = meta.dataset_size.toLocaleString();
                
                // Render feature importance distribution dynamically
                const chartList = document.getElementById('feature-importance-list');
                if (chartList) {
                    chartList.innerHTML = '';
                    
                    // Style config for feature labels and colors matching design spec
                    const featureLabels = {
                        safety: { label: "Safety Rating", color: "#10b981" },
                        persons: { label: "Persons Capacity", color: "#06b6d4" },
                        buying: { label: "Buying Price", color: "#3b82f6" },
                        maint: { label: "Maintenance Cost", color: "#6366f1" },
                        lug_boot: { label: "Luggage Boot Size", color: "#8b5cf6" },
                        doors: { label: "Doors Count", color: "#a855f7" }
                    };

                    // Sort features by their relative importances
                    const sortedFeatures = Object.entries(meta.feature_importances)
                        .sort((a, b) => b[1] - a[1]);

                    sortedFeatures.forEach(([feature, importance]) => {
                        const pct = (importance * 100).toFixed(1);
                        const config = featureLabels[feature] || { label: feature.toUpperCase(), color: '#3b82f6' };
                        
                        const item = document.createElement('div');
                        item.className = 'feature-bar-item';
                        item.innerHTML = `
                            <div class="feature-bar-meta">
                                <span>${config.label}</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width: ${pct}%; background: ${config.color};"></div>
                            </div>
                        `;
                        chartList.appendChild(item);
                    });
                }
            }
        } catch (error) {
            console.error("Error loading model metadata:", error);
        }
    }
});
