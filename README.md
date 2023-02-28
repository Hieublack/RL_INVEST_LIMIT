## :dart: Báo cáo Ticket To Ride 
1.   `Tốc độ chạy`
      - **1000 Game**: 
      - **1000 Game full numba**: 
      - **10000 Game**: 

2. `Chuẩn form`: **Đã test**
3. `Đúng luật`: **Đã check**
4. `Không bị loop vô hạn`: **Đã test** với 1000000 ván
5. `Tốc độ chạy các hàm con mà người chơi dùng`:
6. `Số ván check_vic > victory_thật`: 
9. `Tối thiểu số lần truyền vào player`: 

## :globe_with_meridians: ENV_state
*   [0:21] **Thông tin của công thức 1**: 16 vị trí đầu là lịch sử lợi nhuận trong 16 quý gần nhất có thể xem được, kế tiếp lần lượt là geomean, hamean profit của công thức1, rank value của công ty không đầu tư theo công thức1, ngưỡng của công thức1, value quý hiện tại của công thức1
*   [21:42] **Thông tin của công thức 2**: 16 vị trí đầu là lịch sử lợi nhuận trong 16 quý gần nhất có thể xem được, kế tiếp lần lượt là geomean, hamean profit của công thức2, rank value của công ty không đầu tư theo công thức2, ngưỡng của công thức2, value quý hiện tại của công thức2
*   [42:44] **Người chơi có thể đầu tư hay không** 1 là có thể đầu tư, 0 là đang ở trạng thái ko thể đầu tư
*   [44:46] **Đếm số quý người chơi đã nằm ở trạng thái ko đầu tư** giá trị nằm trong range(0,4)
*   [46:48] **Lợi nhuận sẽ được tính cho người chơi khi người chơi hết chu kì đầu tư** 
*   [48:50] **Tích profit các lần đầu tư của người chơi** 
*   [50] **Vị trí kiểm tra đã hết game hay chưa**
*   [51] **Đếm số người đã action (khi tất cả người chơi đều action thì chuyển quý)**
*   [52] **Vị trí lưu trữ quý hiện tại là quý bao nhiêu**
*   [53:55] **Người chơi đã cập nhật lợi nhuận chưa, nếu rồi thì là 1 còn chưa thì là 0** 
*   [55] **index người chơi action hiện tại**
*   [56:] **Thông tin của hai công thức, mỗi công thức gồm thông tin của công thức gồm (profit các quý, value các quý, rank not invest các quý, ngưỡng các quý**Đoạn này độ dài linh hoạt thay đổi theo số lượng quý của dữ liệu


**Total env_state length: hiện tại là 552**
## :bust_in_silhouette: P_state
*   [0:21] **Thông tin của công thức 1**: 16 vị trí đầu là lịch sử lợi nhuận trong 16 quý gần nhất có thể xem được, kế tiếp lần lượt là geomean, hamean profit của công thức1, rank value của công ty không đầu tư theo công thức1, ngưỡng của công thức1, value quý hiện tại của công thức1
*   [21:42] **Thông tin của công thức 2**: 16 vị trí đầu là lịch sử lợi nhuận trong 16 quý gần nhất có thể xem được, kế tiếp lần lượt là geomean, hamean profit của công thức2, rank value của công ty không đầu tư theo công thức2, ngưỡng của công thức2, value quý hiện tại của công thức2*   
*   [42:44] **Tích profit của các người chơi**
*   [44]    **Người chơi có thể đầu tư hay không (0 hoặc 1)**
*   [45:46] **Các người chơi đã cập nhật lợi nhuận hay chưa**
*   [44]    **Game đã kết thúc hay chưa (0 or 1)**



## :video_game: ALL_ACTION
* action 0: không đầu tư
* action 1: đầu tư theo công thức 1
* action 2: đầu tư theo công thức 2

**Total 3 action**