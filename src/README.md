# Visual Localization Simulation Workflow

Bu dokuman `simulation_26_4_26` dataseti icin kullandigimiz known-pose reconstruction, batch query localization, heading/yaw duzeltmesi ve fake-test deneylerini anlatir.

Ana fikir:

- `collect_multicam_hloc_v2.py` CARLA pozlarini COLMAP uyumlu `qw qx qy qz tx ty tz` formatina cevirir.
- `reconstruction_known_pose.ipynb` kamera pozlarini metadata'dan sabit olarak alir ve sadece 3D point'leri triangulate eder.
- `query_batch_known_pose_v2.ipynb` query localization yapar ve COLMAP rotation sonucunu CARLA yaw frame'ine cevirerek dogru heading error hesaplar.
- `query_batch_known_pose_v2_fake_test.ipynb` gercek ve sahte query gorsellerini karisik test etmek icindir.

## Ortam

Komutlar `VisualLocalization/src` icinden calistirilacak sekilde dusunulmustur.

```powershell
cd C:\Users\ilker\Desktop\bitirme-project\VisualLocalization\src
..\..\venv38\Scripts\activate
```

Notebook'lari VS Code/Jupyter icinden `venv38` kernel'i ile calistir.

## Dataset Yapisi

Ana dataset:

```text
VisualLocalization/datasets/simulation_26_4_26/
  images/                     # mapping/train images, su an front images
  test/                       # query/test images, su an back images
  metadata/
    intrinsics_pinhole.json
    poses.json
    poses_detailed.json
```

`poses.json` basit COLMAP uyumlu pose kayitlarini tutar:

```text
qw qx qy qz tx ty tz
```

Bu degerler COLMAP `images.txt` icin world-to-camera extrinsic degerleridir:

```text
x_camera = R_world_to_camera * x_world + t_world_to_camera
```

Yani `tx ty tz` fiziksel kamera konumu degildir. Kamera merkezini bulmak icin:

```python
C = -R.T @ t
```

`poses_detailed.json` daha acik debug bilgilerini tutar:

- CARLA left-handed camera-to-world matrix
- converted right-handed camera-to-world matrix
- converted right-handed world-to-camera matrix
- quaternion
- translation
- conversion matrix `S`

## CARLA -> COLMAP Donusumu

`collect_multicam_hloc_v2.py` icinde CARLA frame'i COLMAP/camera-style right-handed frame'e su matrisle cevrilir:

```python
S = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])
```

CARLA eksenleri:

```text
x = forward
y = right
z = up
```

Converted/COLMAP-style eksenler:

```text
x = right
y = down
z = forward
```

Capture script su islemi yapar:

```python
R_cw_rh = S @ R_cw_lh @ inv(S)
p_cw_rh = S @ p_cw_lh

R_wc_rh = R_cw_rh.T
t_wc_rh = -R_wc_rh @ p_cw_rh
q_wxyz = rotation_matrix_to_quaternion(R_wc_rh)
```

Sonra `poses.json` icine:

```text
qw qx qy qz tx ty tz
```

olarak `R_wc_rh` ve `t_wc_rh` yazilir. Bunlar COLMAP `images.txt` formatina uygundur.

## Known-Pose Reconstruction

Notebook:

```text
reconstruction_known_pose.ipynb
```

Bu notebook image-only SfM yapmaz. Kamera pozlarini tahmin etmez. Kamera pozlari metadata'dan gelir.

Akis:

1. `simulation_26_4_26/images` altindaki mapping image'lari bundle icine kopyalanir.
2. SuperPoint local features cikarilir.
3. NetVLAD global retrieval features cikarilir.
4. Image pair listesi olusturulur.
5. LightGlue ile feature matching yapilir.
6. `write_known_pose_text_model(...)` calisir.
7. Metadata'daki `qw qx qy qz tx ty tz` degerleri `sparse_input/images.txt` icine yazilir.
8. `triangulation.main(...)` bu known-pose model'i `reference_model` olarak kullanir.
9. COLMAP database olusturulur ve 3D point'ler triangulate edilir.

En kritik iki nokta:

```python
write_known_pose_text_model(...)
```

Bu fonksiyon metadata'daki quaternion ve translation degerlerini COLMAP text model'e yazar:

```text
sparse_input/cameras.txt
sparse_input/images.txt
sparse_input/points3D.txt
```

`points3D.txt` basta bostur. Yani bu asamada sadece kamera pose skeleton'i vardir.

```python
triangulation.main(
    reference_model=sparse_input_dir,
    ...
)
```

Bu adim known camera pose'lari sabit kabul edip feature match ray'lerinden 3D point uretir.

Beklenen output:

```text
VisualLocalization/outputs/simulation-known-pose-bundle/
  images/
  sparse_input/
  sfm/
  sparse/
  features.h5
  global-feats-netvlad.h5
  matches.h5
  pairs-sfm.txt
  metadata.json
```

Bizim son calismada known-pose map yaklasik su sonucu verdi:

```text
registered images: 203
points3D: ~89514
mean reprojection error: ~1.10 px
metadata camera center alignment error: ~0 m
```

Bu alignment error'un 0'a yakin olmasi normaldir, cunku kamera pozlari zaten metadata'dan veriliyor.

## Neden Image-Only Reconstruction Kotu Sonuc Vermisti?

Eski `reconstruction.ipynb` sadece image input kullanarak SfM yapiyordu. CARLA metadata pose'lari reconstruction'a dahil edilmiyordu.

Bu nedenle:

- COLMAP kendi scale/orientation/frame'ini tahmin ediyordu.
- Uzun ve benzer yol sahnelerinde drift/deformation olusabiliyordu.
- Global Sim(3) alignment yaptigimizda mean error yaklasik 20 m seviyesine cikmisti.

Bu metadata'nin yanlis oldugu anlamina gelmiyordu. Sorun image-only SfM map'in ground-truth frame'e elastik/driftli sekilde oturmasiydi.

Known-pose reconstruction bu problemi cozdu:

- Kamera merkezleri metadata ile ayni frame'de kaldi.
- Sadece 3D noktalar triangulate edildi.
- Query position error cok dusuk hale geldi.

## Batch Query Localization v2

Notebook:

```text
query_batch_known_pose_v2.ipynb
```

Varsayilan ayarlar:

```python
dataset_root = Path('../datasets/simulation_26_4_26')
query_source_dir = dataset_root / 'test'
max_queries = 50
random_seed = 42

map_bundle_root = Path('../outputs/simulation-known-pose-bundle')
sfm_model_root = map_bundle_root / 'sfm'

results_dir = map_bundle_root / 'query_batch_results_v2'
query_cache_dir = map_bundle_root / 'query_batch_v2'
```

Calisma sirasi:

1. Once `reconstruction_known_pose.ipynb` calistirilmis olmali.
2. Sonra `query_batch_known_pose_v2.ipynb` calistirilir.
3. Query image'lar `test/` klasorunden secilir.
4. Her query icin NetVLAD retrieval yapilir.
5. En yakin `num_loc` map image ile SuperPoint + LightGlue matching yapilir.
6. HLoc/PnP ile query pose estimate edilir.
7. Estimated camera center ground truth ile karsilastirilir.
8. Estimated heading, COLMAP frame'den CARLA yaw frame'ine cevrilip karsilastirilir.

Onemli parametreler:

```python
num_loc = 10
```

Her query icin retrieval'dan kac reference image kullanilacagini belirler.

```python
max_error = 12
```

PnP/RANSAC reprojection error threshold'udur. Piksel cinsindendir.

```python
overwrite_query_features = True
```

Query feature/match dosyalarini yeniden uretir. Ayni query setini tekrar tekrar test ederken `False` yapmak hiz kazandirir.

## Heading/Yaw Hatasinin Sebebi

Ilk query notebook'ta heading su sekilde hesaplanmisti:

```python
R.from_matrix(R_wc_rh).as_euler('xyz', degrees=True)[2]
```

Bu yanlisti, cunku:

- PnP sonucu COLMAP/right-handed world-to-camera rotation verir.
- `poses.json["yaw_deg"]` CARLA left-handed yaw degeridir.
- Bu iki aci direkt karsilastirilamaz.

Bu yuzden heading error yaklasik 16 derece cikiyordu.

## Heading/Yaw Duzeltmesi

v2 notebook'ta estimated rotation once CARLA left-handed camera-to-world frame'e geri cevriliyor:

```python
CARLA_TO_COLMAP_S = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])

COLMAP_TO_CARLA_S = np.linalg.inv(CARLA_TO_COLMAP_S)

def colmap_world_to_camera_to_carla_yaw_deg(r_wc_rh):
    r_cw_rh = np.asarray(r_wc_rh, dtype=np.float64).T
    r_cw_lh = COLMAP_TO_CARLA_S @ r_cw_rh @ CARLA_TO_COLMAP_S
    return wrap_angle_deg(np.degrees(np.arctan2(r_cw_lh[1, 0], r_cw_lh[0, 0])))
```

Bu donusumden sonra `poses.json["yaw_deg"]` ile karsilastirma dogru frame'de yapiliyor.

Metadata sanity check:

```python
metadata_record_to_carla_yaw_deg(record)
```

Bu fonksiyon `poses.json` quaternion'larini geri CARLA yaw'a cevirir. Beklenen error:

```text
max error < 1e-3 deg
```

Bizim testte:

```text
mean error: ~0.000002 deg
max error: ~0.000007 deg
```

Bu, quaternion/matrix conversion'in tutarli oldugunu gosterir.

v2 sonuc:

```text
total queries: 50
successful: 50
position error mean/median/max: 0.0196 / 0.0081 / 0.1162 m
heading error mean/median/max: 0.0252 / 0.0097 / 0.1319 deg
```

Threshold:

```text
All conditions: (0.25m, 2 deg) / (0.5m, 5 deg) / (5m, 10 deg)
All conditions: 100.0 / 100.0 / 100.0
```

Bu sonuc simule, kucuk, ayni rota/kosul dataseti icin normaldir. Gercek genelleme iddiasi degildir.

## Fake Test Deneyi

Dataset:

```text
VisualLocalization/datasets/fake_test/
  test/
  metadata/
    intrinsics_pinhole.json
    poses.json
    poses_detailed.json
    fake_test_manifest.json
```

Notebook:

```text
query_batch_known_pose_v2_fake_test.ipynb
```

Bu deneyde:

- 10 tane gercek `simulation_26_4_26/test` image'i kullanildi.
- 10 tane internetten rastgele image kullanildi.
- Gercek image'lar kendi `poses.json` kayitlarini korudu.
- Random image'lar icin model frame'iyle uyumlu olsun diye dataset'ten rastgele q/t/matrix kayitlari kopyalandi ve synthetic capture id verildi.

Bu random image pose'lari gercek ground truth degildir. Sadece evaluation pipeline'in fake image'lari nasil eleyecegini gormek icindir.

Fake-test sonuc:

```text
total queries: 20
real back images: 10/10 threshold icinde
fake random images: huge position/heading error
All conditions: 50.0 / 50.0 / 50.0
```

Bu beklenen davranistir:

- Gercek 10 query basarili.
- Random 10 query dogru lokalize olmuyor.
- Bazi random query'ler `success=True` olabilir. Bu sadece PnP'nin bir pose dondurdugu anlamina gelir.
- Dogru lokalizasyon sayilmasi icin threshold'u gecmesi gerekir.

## Bilinen Problemler ve Notlar

### `translation_xyz_m` altitude degildir

`tx ty tz` COLMAP world-to-camera translation'dir. Kamera konumu degildir.

Kamera merkezi:

```python
C = -R.T @ t
```

Rampadaki yukseklik degisimi icin `tz` yerine camera center veya `matrix_camera_to_world[:3, 3]` incelenmelidir.

### Heading error ilk basta 16 dereceydi

Bu pose estimation hatasi degildi. COLMAP/right-handed Euler yaw ile CARLA `yaw_deg` direkt karsilastiriliyordu.

Duzeltme:

```text
COLMAP world-to-camera rotation
-> COLMAP camera-to-world rotation
-> CARLA left-handed camera-to-world rotation
-> CARLA yaw
```

### Known-pose map alignment error 0 cikiyor

Bu normaldir. Kamera pozlari zaten metadata'dan sabit veriliyor.

Bu sonuc map'in goruntu bazli genellestigini degil, metadata frame ile tutarli oldugunu gosterir.

### HDF5 file locking / Windows match_features problemi

Fake-test notebook'unda Windows uzerinde HDF5 file locking hatalari gorulebildi. Bunun icin notebook'ta HLoc writer queue tek yaziciya dusuruldu:

```python
match_features.WorkQueue = _SingleWriterWorkQueue
```

Bu fake-test stress deneyi icin daha stabil calisir.

### `overwrite_query_features`

Ayni query setini tekrar calistirirken:

```python
overwrite_query_features = False
```

daha hizlidir.

Feature extractor ayarlari degisirse veya temiz deneme istenirse:

```python
overwrite_query_features = True
```

kullanilmalidir.

## Onerilen Calisma Sirasi

1. Dataset topla:

```text
collect_multicam_hloc_v2.py
```

2. Known-pose map olustur:

```text
reconstruction_known_pose.ipynb
```

3. Normal query test yap:

```text
query_batch_known_pose_v2.ipynb
```

4. Fake/real mixed test yap:

```text
query_batch_known_pose_v2_fake_test.ipynb
```

5. Sonuclari incele:

```text
outputs/simulation-known-pose-bundle/query_batch_results_v2/
outputs/simulation-known-pose-bundle/query_batch_results_fake_test_v2/
```

## Sonuc Yorumlama

`success=True` tek basina dogru lokalizasyon demek degildir. Sadece PnP'nin pose dondurdugunu gosterir.

Dogru lokalizasyon icin threshold sonucuna bak:

```text
All conditions: (0.25m, 2 deg) / (0.5m, 5 deg) / (5m, 10 deg)
```

Kucuk ve kontrollu simulation datasinda 100% sonuc normal olabilir. Daha guclu test icin:

- farkli rota
- farkli hava/isik
- farkli kamera
- daha uzun trajectory
- gercek negatif/fake query setleri

kullanilmalidir.
