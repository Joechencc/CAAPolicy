```markdown
# File Transfer using rclone

This guide walks through setting up rclone to transfer files to/from Google Drive.

## Step 1: Install rclone & Configuration

1. Install rclone from [https://rclone.org/install/](https://rclone.org/install/)

2. Start the configuration tool:
   ```bash
   rclone config
   ```

3. You will see this output:
   ```
   No remotes found - make a new one
   n) New remote
   s) Set configuration password
   q) Quit config
   n/s/q> 
   ```
   Type `n` to create a New remote and enter a name when prompted (e.g., `mygoogledrive`):
   ```
   name> mygoogledrive
   ```

4. Select Google Drive from the storage options:
   ```
   Type of storage to configure.
   Choose a number from below, or type in your own value
   ...
   9 / Google Drive
      \ "drive"
   Storage> 9
   ```

5. Leave the next two prompts blank (client_id and client_secret) by just pressing Enter:
   ```
   Google Application Client Id - leave blank normally.
   client_id>  
   Google Application Client Secret - leave blank normally.
   client_secret> 
   ```

6. For auto config, select `n` (since you're working on a remote/headless machine):
   ```
   Remote config
   Use auto config?
    * Say Y if not sure
    * Say N if you are working on a remote or headless machine or Y didn't work
   y) Yes
   n) No
   y/n> n
   ```

7. You'll see a verification URL. Either:
   - Your browser will open automatically, OR
   - Manually navigate to the link in your browser and click "Allow Access"

8. Paste the verification code back in the terminal:
   ```
   Enter verification code> YOURCODE
   ```

9. When asked about team drive, select `n`:
   ```
   Configure this as a team drive?
   y) Yes
   n) No
   y/n> n
   ```

10. Confirm the configuration is correct:
    ```
    y) Yes this is OK
    e) Edit this remote
    d) Delete this remote
    y/e/d> y
    ```

11. Quit the config tool:
    ```
    q) Quit config
    e/n/d/r/c/s/q> q
    ```

## Step 2: Copy Files

Use the following command to copy files:
```bash
rclone copy {SOURCE_PATH} {DEST_PATH}
```

Example:
```bash
rclone copy AI4CE:E2E-Bosch/parkinglot_1000 /vast/rs9193/Town_Opt_1000
```

For more information, visit: [https://noisyneuron.github.io/nyu-hpc/transfer.html](https://noisyneuron.github.io/nyu-hpc/transfer.html)
```
