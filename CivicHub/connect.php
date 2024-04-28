<?php
     $name = $_POST['name'];
     $email = $_POST['email'];
     $phone = $_POST['phone'];
     $age = $_POST['age'];
     $gender = $_POST['gender'];

     //Database connection
     $conn = new mysqli('localhost', 'root','','proj');
     if($conn->connect_error){
        die('Connection Failed :'.$conn->connect_error);
     }else{
        $stmt = $conn->prepare("insert into registration(name,email,phone,age,gender)
            values(?,?,?,?,?)");
        $stmt->bind_param("ssiis",$name,$email,$phone,$age,$gender);
        $stmt->execute();
        echo "registration sucessfully...";
        $stmt->close();
        $conn->close();    
     }
?>
