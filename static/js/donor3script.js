function addItem() {
    let name = document.getElementById("name").value;
    let quantity = document.getElementById("quantity").value;
    let date = document.getElementById("date").value;
    let imageInput = document.getElementById("image");

    if (!name || !quantity || !date || !imageInput.files.length) {
        alert("Please fill in all fields and upload an image.");
        return;
    }

    let cart = document.getElementById("cart");
    let cartItems = cart.getElementsByClassName("cart-item");

    if (cartItems.length >= 5) {
        alert("You can only add up to 5 items.");
        return;
    }

    let item = document.createElement("div");
    item.classList.add("cart-item");

    let reader = new FileReader();
    reader.onload = function(event) {
        let imageSrc = event.target.result;

        item.innerHTML = `
            <img src="${imageSrc}" alt="Item Image">
            <p><strong>Name:</strong> ${name}</p>
            <p><strong>Quantity:</strong> ${quantity} kg</p>
            <p><strong>Made on:</strong> ${date}</p>
            <button class="remove-btn" onclick="removeItem(this)">Remove</button>
        `;

        cart.appendChild(item);

        // Clear input fields
        document.getElementById("name").value = "";
        document.getElementById("quantity").value = "";
        document.getElementById("date").value = "";
        document.getElementById("image").value = "";
    };

    reader.readAsDataURL(imageInput.files[0]); // Convert image to data URL
}

function removeItem(button) {
    button.parentElement.remove();
}

function predictItems() {
    alert("Prediction feature is under development!");
}
