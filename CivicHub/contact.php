<?php
     $name = $_POST['name'];
     $email = $_POST['email'];
     $reason = $_POST['reason'];
     $detail = $_POST['detail'];

     //Database connection
     $conn = new mysqli('localhost', 'root','','proj');
     if($conn->connect_error){
        die('Connection Failed :'.$conn->connect_error);
     }else{
        $stmt = $conn->prepare("insert into contact(name,email,reason,detail)
            values(?,?,?,?)");
        $stmt->bind_param("ssss",$name,$email,$reason,$detail);
        $stmt->execute();
        echo "submitted sucessfully...";
        $stmt->close();
        $conn->close();    
     }
?>
