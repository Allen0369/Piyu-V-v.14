function fileValidation() {
	var puvCapacity = document.getElementById('puv_capacity').value;
	var puvUnits = document.getElementById('puv_units').value;

	if (
		!puvCapacity || isNaN(puvCapacity) || puvCapacity % 1 !== 0 || puvCapacity <= 0 ||
		!puvUnits || isNaN(puvUnits) || puvUnits % 1 !== 0 || puvUnits <= 0
	) {
		alert("Please enter a number greater than 0 for both capacity and units before uploading.");
		document.getElementById('entry_value').value = null;
		return false;
	}
	
	var fileInput = document.getElementById('entry_value');
	var files = fileInput.files;
	
	var allowedExtensions = /(\.csv)$/i;
	var requiredHeaders = ['date', 'PULocationID', 'DOLocationID', 'day_of_week', 'passenger_count', 'trip_distance', 'trip_duration'];
	
	let processedCount = 0;
	let allValid = true;

	for (let i = 0; i < files.length; i++) {
		const file = files[i];
		const fileName = file.name.toLowerCase();

		if (!allowedExtensions.exec(fileName)) {
			alert(`Invalid file type. Only .csv files are allowed.`);
			fileInput.value = '';
			return false;
		}

		var reader = new FileReader();

		reader.onload = function (e) {
			let headers;

			if (fileName.endsWith('.csv')) {
				const text = e.target.result;
				const lines = text.trim().split('\n');
				headers = lines[0].split(',').map(h => h.trim());

				const missing = requiredHeaders.filter(h => !headers.includes(h));
				if (missing.length > 0) {
					alert(`File "${file.name}" is missing column(s): ${missing.join(', ')}`);
					fileInput.value = '';
					allValid = false;
					return;
				}
			}

			processedCount++;
			if (processedCount === files.length && allValid) {
				sendToFlask(files);
			}
		};

		if (fileName.endsWith('.csv')) {
			reader.readAsText(file);
		}
	}
}

function sendToFlask(files) {
	const formData = new FormData();
	for (let i = 0; i < files.length; i++) {
		formData.append("files", files[i]);
	}
	
	const puvCapacity = document.getElementById("puv_capacity").value;
	const puvUnits = document.getElementById("puv_units").value;
	formData.append("capacity", puvCapacity);
	formData.append("units", puvUnits);

	const progressBarContainer = document.getElementById("progressBarContainer");
	const progressBar = document.getElementById("progressBar");
	progressBarContainer.style.display = "block";
	progressBar.style.width = "0%";

	let fakeProgress = 0;
	const interval = setInterval(() => {
		if (fakeProgress < 95) {
			fakeProgress += Math.random() * 5;
			progressBar.style.width = fakeProgress + "%";
		}
	}, 300);

	fetch("/predict", {
		method: "POST",
		body: formData
	})
	.then(response => response.json())
	.then(data => {
		clearInterval(interval);
		progressBar.style.width = "100%";

		if (!data.session_id) {
			alert("Error: No session ID received from server.");
			console.error("Response from server:", data);
			progressBarContainer.style.display = "none";
			progressBar.style.width = "0%";
			return;
		}
		setTimeout(() => {
			progressBarContainer.style.display = "none";
			progressBar.style.width = "0%";
			window.location.href = `/dashboard?session_id=${data.session_id}`;
		}, 500);
	})
}

function on_upload() {
	document.getElementById("overlay_upload").style.display = "block";
}


function off_upload() {
    const modal = document.getElementById('text_upload');
    if (!modal.contains(event.target)) {
        document.getElementById('overlay_upload').style.display = "none";
        const radios = document.getElementsByName('puv_type');
        radios.forEach(radio => radio.checked = false);
    }
}

/*
function on_save() {
	const elementToCapture = document.getElementById("results");

	const progressBarContainer = document.getElementById("progressBarContainer");
	const progressBar = document.getElementById("progressBar");
	progressBarContainer.style.display = "block";
	progressBar.style.width = "0%";

	let fakeProgress = 0;
	const interval = setInterval(() => {
		if (fakeProgress < 95) {
			fakeProgress += Math.random() * 5;
			progressBar.style.width = fakeProgress + "%";
		}
	}, 200);

	html2canvas(elementToCapture).then(canvas => {
		clearInterval(interval);
		progressBar.style.width = "100%";

		const imageData = canvas.toDataURL("image/png");
		const downloadLink = document.createElement("a");
		downloadLink.href = imageData;
		downloadLink.download = "Piyu-V_Report.png";
		downloadLink.click();

		setTimeout(() => {
			progressBarContainer.style.display = "none";
			progressBar.style.width = "0%";

			document.getElementById("overlay_save").style.display = "block";
			setTimeout(() => {
				document.getElementById("overlay_save").style.display = "none";
			}, 2000);
		}, 500);

	}).catch(error => {
		clearInterval(interval);
		progressBarContainer.style.display = "none";
		progressBar.style.width = "0%";
		alert("Error saving report: " + error);
	});
}*/