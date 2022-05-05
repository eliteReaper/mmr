import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/material.dart';
import 'package:flutter/material.dart';
import 'package:mmr_app/screens/Pages/account_page.dart';

import '../models/user.dart';

class RatedDataPage extends StatefulWidget {
  const RatedDataPage({Key? key}) : super(key: key);

  @override
  State<RatedDataPage> createState() => _RatedDataPageState();
}

class _RatedDataPageState extends State<RatedDataPage> {
  List<CustomUser> list_user = [];
  @override
  Widget build(BuildContext context) {
    List<CustomUser> user_list = [];
    DatabaseReference ref = FirebaseDatabase.instance.ref('movies');
    var user = (FirebaseAuth.instance.currentUser!.uid.toString());
    // user_list.clear();
    ref.child(user).once().then(
      (snapshot) {
        Map<String, String> mp = Map();
        for (var snap in snapshot.snapshot.children) {
          // print(snap.key.toString());
          // print(snap.value.toString());
          for (var child in snap.children) {
            // print(child.key! + " -- " + child.value.toString());
            mp[child.key!] = child.value.toString();
          }
          String? name = mp['name'];
          String? rating = mp['rating'];
          String? time = mp['time'];
          user_list.add(CustomUser(user, name!, rating!, time!));
        }
      },
    ).whenComplete(() {
      setState(() {
        list_user = user_list;
      });
    });
    if (list_user.length == 0) {
      return Scaffold(
        appBar: AppBar(
          title: Text("Rated media"),
        ),
        body: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    } else {
      print(list_user.length);
      return Scaffold(
        appBar: AppBar(
          title: Text("Rated media"),
        ),
        body: ListView.builder(
          itemCount: list_user.length,
          itemBuilder: (BuildContext context, int index) {
            return Card(
              elevation: 5,
              child: Container(
                padding: const EdgeInsets.all(5),
                margin: const EdgeInsets.all(5),
                decoration: const BoxDecoration(
                  borderRadius: BorderRadius.all(
                    Radius.circular(10),
                  ),
                ),
                child: Row(
                  children: [
                    Expanded(child: Text((index + 1).toString() + ". " + list_user[index].name)),
                    
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        Text(list_user[index].rating),
                        SizedBox(width: 10,),
                        Icon(Icons.star),
                      ],
                    )
                  ],
                ),
              ),
            );
          },
        ),
      );
    }
  }
}
