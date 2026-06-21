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

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || 'Server error occurred');
            }

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

        // Display outcome descriptive copy
        const descriptions = {
            unacc: "The vehicle exhibits highly critical limitations (most often related to unsafe ratings or disproportionate maintenance prices), making it unacceptable for buying or standard operations.",
            acc: "The vehicle satisfies core requirements. While some features may be moderate or basic, it is deemed acceptable for routine tasks.",
            good: "The vehicle presents a reliable quality configuration. Features like high safety paired with reasonable cost models make it a good selection.",
            vgood: "This vehicle is in outstanding condition, matching the highest safety rankings combined with low costs or very high capacity. It rates as very good."
        };
        explanationText.textContent = descriptions[prediction] || "Vehicular parameters processed successfully.";
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
});
