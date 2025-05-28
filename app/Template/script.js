// Configuration - Update this path to your JSON file
const DATA_FILE_PATH = 'report.json';

let analysisData = null;

async function loadData() {
    try {
        const response = await fetch(DATA_FILE_PATH);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        analysisData = await response.json();
        renderDashboard();
    } catch (error) {
        console.error('Error loading data:', error);
        showError();
    }
}

function showError() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'block';
}

function renderDashboard() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
    document.getElementById('tabNavigation').style.display = 'block';

    renderOverallStats();
    renderThresholds();
    renderClassTable();
    renderTopBottom();
    renderPlots();
    checkPredictionImages();
}

function renderOverallStats() {
    const stats = analysisData.summary.overall;
    const container = document.getElementById('overallStats');
    
    const statsData = [
        { label: 'Total Detections', value: stats.total_detections.toLocaleString(), icon: '🎯' },
        { label: 'Mean Confidence', value: (stats.mean_confidence * 100).toFixed(2) + '%', icon: '📊' },
        { label: 'Median Confidence', value: (stats.median_confidence * 100).toFixed(2) + '%', icon: '📈' },
        { label: 'Standard Deviation', value: (stats.std_deviation * 100).toFixed(2) + '%', icon: '📉' },
        { label: 'Min Score', value: (stats.min_score * 100).toFixed(2) + '%', icon: '⬇️' },
        { label: 'Max Score', value: (stats.max_score * 100).toFixed(2) + '%', icon: '⬆️' }
    ];

    container.innerHTML = statsData.map(stat => `
        <div class="stat-card">
            <div class="stat-value">${stat.icon} ${stat.value}</div>
            <div class="stat-label">${stat.label}</div>
        </div>
    `).join('');
}

function renderThresholds() {
    const thresholds = analysisData.summary.confidence_thresholds;
    const container = document.getElementById('thresholdGrid');
    
    container.innerHTML = Object.entries(thresholds).map(([threshold, data]) => `
        <div class="threshold-card">
            <div class="threshold-label">${threshold} Confidence</div>
            <div class="threshold-value">${data.count.toLocaleString()}</div>
            <div class="threshold-percentage">${data.percentage.toFixed(1)}% of total</div>
        </div>
    `).join('');
}

function renderClassTable() {
    const classes = analysisData.summary.class_performance_rankings;
    const tbody = document.getElementById('classTableBody');
    
    // Sort classes by mean confidence
    const sortedClasses = Object.entries(classes).sort((a, b) => b[1].mean - a[1].mean);
    
    tbody.innerHTML = sortedClasses.map(([className, data]) => {
        const isTopPerformer = analysisData.summary.top_5_best_performing_classes.hasOwnProperty(className);
        const isBottomPerformer = analysisData.summary.bottom_5_performing_classes.hasOwnProperty(className);
        
        return `
            <tr class="${isTopPerformer ? 'top-performing' : isBottomPerformer ? 'bottom-performing' : ''}">
                <td class="class-name">${className}</td>
                <td>${data.count.toLocaleString()}</td>
                <td>${(data.mean * 100).toFixed(2)}%</td>
                <td>${isNaN(data.std) ? 'N/A' : (data.std * 100).toFixed(3) + '%'}</td>
                <td>
                    <div class="performance-bar">
                        <div class="performance-fill" style="width: ${data.mean * 100}%"></div>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

function renderTopBottom() {
    const topPerformers = analysisData.summary.top_5_best_performing_classes;
    const bottomPerformers = analysisData.summary.bottom_5_performing_classes;
    
    // Render top performers
    const topTable = document.getElementById('topPerformers');
    topTable.innerHTML = `
        <thead>
            <tr><th>Class</th><th>Confidence</th></tr>
        </thead>
        <tbody>
            ${Object.entries(topPerformers).map(([className, confidence]) => `
                <tr class="top-performing">
                    <td class="class-name">${className}</td>
                    <td>${(confidence * 100).toFixed(2)}%</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    // Render bottom performers
    const bottomTable = document.getElementById('bottomPerformers');
    bottomTable.innerHTML = `
        <thead>
            <tr><th>Class</th><th>Confidence</th></tr>
        </thead>
        <tbody>
            ${Object.entries(bottomPerformers).map(([className, confidence]) => `
                <tr class="bottom-performing">
                    <td class="class-name">${className}</td>
                    <td>${(confidence * 100).toFixed(2)}%</td>
                </tr>
            `).join('')}
        </tbody>
    `;
}

function renderPlots() {
    const plots = analysisData.plots;
    const container = document.getElementById('plotsGrid');
    
    const plotDescriptions = {
        'confidence_distribution': 'Distribution of confidence scores across all detections',
        'class_performance': 'Performance comparison between different object classes',
        'class_distribution': 'Number of detections per object class',
        'confidence_detailed': 'Detailed confidence analysis with statistical measures',
        'rainbow_summary': 'Comprehensive visual summary of all metrics'
    };
    
    container.innerHTML = Object.entries(plots).map(([plotName, plotPath]) => {
        const displayName = plotName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const description = plotDescriptions[plotName] || 'Analysis visualization';
        
        return `
            <div class="plot-card">
                <div class="plot-header">
                    <div class="plot-title">${displayName}</div>
                    <div class="plot-description">${description}</div>
                </div>
                <img src="${plotPath}" alt="${displayName}" class="plot-image" 
                     onclick="openImageModal('${plotPath}', '${displayName}', '${description}')"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                <div class="plot-placeholder" style="display: none;">
                    Image not available: ${plotPath}
                </div>
            </div>
        `;
    }).join('');
}

// Image Modal Functions
function openImageModal(imageSrc, title, description) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    
    modalImage.src = imageSrc;
    modalCaption.textContent = `${title} - ${description}`;
    
    modal.style.display = 'flex';
    // Trigger animation after display is set
    setTimeout(() => {
        modal.classList.add('show');
    }, 10);
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('show');
    
    // Hide modal after animation completes
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
}

// Event Listeners for Modal
document.addEventListener('DOMContentLoaded', function() {
    // Close modal when clicking the close button
    document.querySelector('.close-button').addEventListener('click', closeImageModal);
    
    // Close modal when clicking outside the image
    document.getElementById('imageModal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeImageModal();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeImageModal();
        }
    });
});

// Tab Switching Functions
function switchTab(tabName) {
    const analysisContent = document.getElementById('dashboard');
    const predictionsContent = document.getElementById('predictionsContent');
    const tabButtons = document.querySelectorAll('.tab-button');
    
    // Remove active class from all buttons
    tabButtons.forEach(button => button.classList.remove('active'));
    
    if (tabName === 'analysis') {
        analysisContent.style.display = 'block';
        predictionsContent.style.display = 'none';
        tabButtons[0].classList.add('active');
    } else if (tabName === 'predictions') {
        analysisContent.style.display = 'none';
        predictionsContent.style.display = 'block';
        tabButtons[1].classList.add('active');
        loadPredictionImages();
    }
}

// Check if prediction images exist and show/hide tab accordingly
async function checkPredictionImages() {
    try {
        // Check if we have generated images information in the analysis data
        if (analysisData && analysisData.has_prediction_images && analysisData.generated_images && analysisData.generated_images.length > 0) {
            // Images were generated, show the predictions tab
            document.getElementById('predictionsTab').style.display = 'block';
        } else {
            // No images were generated, hide the predictions tab
            document.getElementById('predictionsTab').style.display = 'none';
        }
    } catch (error) {
        // Error checking for images, hide the predictions tab
        document.getElementById('predictionsTab').style.display = 'none';
    }
}

// Load and display prediction images
async function loadPredictionImages() {
    const container = document.getElementById('predictionsGrid');
    const noPredictions = document.getElementById('noPredictions');
    
    try {
        // Use the generated_images list from the analysis data
        if (analysisData && analysisData.generated_images && analysisData.generated_images.length > 0) {
            const existingImages = [];
            
            // Verify that the images actually exist by trying to load them
            for (const imageName of analysisData.generated_images) {
                try {
                    const response = await fetch(`images/${imageName}`, { method: 'HEAD' });
                    if (response.ok) {
                        existingImages.push(imageName);
                    }
                } catch (e) {
                    // Image doesn't exist or can't be accessed, skip it
                    console.warn(`Could not access image: ${imageName}`);
                }
            }
            
            if (existingImages.length === 0) {
                // Show no predictions message
                showNoPredictionsMessage(container);
            } else {
                // Show images
                renderPredictionImages(existingImages);
            }
        } else {
            // No generated images in the data
            showNoPredictionsMessage(container);
        }
        
    } catch (error) {
        console.error('Error loading prediction images:', error);
        showErrorMessage(container);
    }
}

// Show no predictions available message
function showNoPredictionsMessage(container) {
    container.innerHTML = `
        <div class="no-predictions">
            <div class="no-predictions-icon">📷</div>
            <h3>No Prediction Images Available</h3>
            <p>Prediction images will appear here when generated using the --generate_fig option.</p>
            <div class="no-predictions-code">
                <code>python app.py --folder /path/to/images --generate_fig True</code>
            </div>
        </div>
    `;
}

// Show error message
function showErrorMessage(container) {
    container.innerHTML = `
        <div class="no-predictions">
            <div class="no-predictions-icon">⚠️</div>
            <h3>Error Loading Images</h3>
            <p>Could not access the images directory.</p>
        </div>
    `;
}

// Render prediction images in the grid
function renderPredictionImages(images) {
    const container = document.getElementById('predictionsGrid');
    
    container.innerHTML = images.map(imageName => {
        const imageBaseName = imageName.replace(/\.[^/.]+$/, ""); // Remove extension
        const displayName = imageBaseName.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1');
        
        return `
            <div class="prediction-card">
                <img src="images/${imageName}" 
                     alt="Prediction for ${displayName}" 
                     class="prediction-image"
                     onclick="openImageModal('images/${imageName}', 'Prediction: ${displayName}', 'Object detection results with bounding boxes and confidence scores')"
                     onerror="this.parentElement.style.display='none';">
                <div class="prediction-info">
                    <div class="prediction-filename">${displayName}</div>
                    <div class="prediction-meta">Click to view full size</div>
                </div>
            </div>
        `;
    }).join('');
}

// Enhanced function to discover images by trying common patterns
async function discoverPredictionImages() {
    const imageExtensions = ['jpg', 'jpeg', 'png'];
    const discoveredImages = [];
    
    // Generate potential image names based on common patterns
    const patterns = [
        // Frame-based patterns
        ...Array.from({length: 100}, (_, i) => `frame${String(i).padStart(4, '0')}_leftImg8bit.jpg`),
        // Image-based patterns  
        ...Array.from({length: 100}, (_, i) => `image_${String(i).padStart(3, '0')}.jpg`),
        ...Array.from({length: 100}, (_, i) => `img_${String(i).padStart(3, '0')}.png`),
    ];
    
    // Test a subset of patterns to avoid too many requests
    const testPatterns = patterns.slice(0, 20);
    
    for (const pattern of testPatterns) {
        try {
            const response = await fetch(`images/${pattern}`, { method: 'HEAD' });
            if (response.ok) {
                discoveredImages.push(pattern);
            }
        } catch (e) {
            // Continue silently
        }
    }
    
    return discoveredImages;
}

// Initialize the dashboard
loadData();