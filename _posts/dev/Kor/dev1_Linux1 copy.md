---
layout: post
image:  /assets/images/blog/post-5.jpg
mathjax: true
title: "[Autonomy devcourse 1$_{st}$] Linux"
last_modified_at: 2023-04-27
categories:
  - Paper review
tags:
  - Autonomy-driving
  - dev
excerpt: "Dev course review"
use_math: true
classes: wide
---

<center><img src='{{"/assets/images/blog/dev/1_linux/Linux-Logo-700x394.png" | relative_url}}' width="50%"></center>

## 1. Introduction
- \- All electronic devices such as computers and smartphones are composed of hardware and software.
   - \- Hardware (HW): everything physically in an electronic device
   - \- Software (SW): A collection of commands given to a computer to achieve a specific purpose
- \- When using the software, the computer's CPU, RAM, and other hardware are used to perform the operation requested by the user.
   - \- At this time, the Operating System (OS) allocates hardware resources as needed for the software.
     - \- The operating system manages limited hardware resources such as CPU and RAM and mediates between HW and SW.
     - ex. Windows, MAC, Android, IOS etc.
- \- Linux is also an operating system
   - \- An operating system based on UNIX by Linus Torvals, a Finnish SW engineer.

## 2. Why use Linux?
- \- **Open source**
   - \- Linux is an open source operating system
   - \- Anyone, whether an individual or a corporation, can install and use Linux for free.
   - \- open source movement

- \- **Customizing**
   - \- Linux strictly means the Linux kernel.
     - \- The kernel is a part of the OS that performs the core functions of the OS.
   - \- Only the core functions that Linux OS needs to perform are defined, and the other parts can be customized and used by the user according to his/her own use.

- \- **Stable operation**
   - \- Since it is open source, various users can verify it in real time.
   
<center>
  <figure>
    <img src='{{"/assets/images/blog/dev/1_linux/Sample Open-source sw.jpg" | relative_url}}' width="100%">
    <figcaption>
      Figure 2. Sample open-source software for IBM Z and LinuxONE. Image source: <a href="https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/javier-perez1/2021/03/30/the-growing-ecosystem-of-open-source-software-for">[IBM community]</a>.
    </figcaption>
  </figure>
</center>


## 3. Linux(UNIX) Required Commands
### 3.1. CLI(Command Line Interface) commands
   - \- Made for i18n (internationalization)
   - \- UTF-8 is used as the default character set
   - \- It is recommended to use the en_US locale as it is affected by the setting of the LANG environment variable

#### File
- \- **Path**
  - \- command
    - $pwd$: print working directory
    - $cd$: change directory
  - \- /: root directory
  - \- ~: home directory
  - \- -: previous directory
  - \- Path type
    - \- Absolute path (abs-path): path starting from the root directory
    - \- Relative path: path starting from the current directory (.)

- \- **Check**
   - \- command
     - \- $ls, file, stat, which, find$
   
   - \- file mode bit: 3+9 bit system representing UNIX file permissions
     - \- how to write
       - \- Symbolic mode:
         - How to mark with "rwx" symbol
         - Consists of owner, group, and others parts, each with 3 spaces
         - r: readable, w: writable, e: executable
       - \- Octal mode: A method of expressing bits in octal notation

    - \- $stat$: outputs status of file, meta data of file
      - \- meta data: Modifying information, not content (file name, creation time, permissions)
    
    - \- $touch$: update meta data of file, create an empty file if file does not exist
    
    - \- $find$: find directory
      - \-name filename: search for files with the same name as filename
      - \-size n: search for files of size n
      - \-mtime n: search for files with modified time n
      - \-inum n: Search for files with inode number n
      - \-max(min)depth level: Search for files with a maximum (minimum) depth of level in the subdirectory of the location to be searched
      - $ex1.$ find . -name '*k.data' -a -size 1M (-a: AND, -o: OR)
      - $ex2.$ find -name "*.tmp" -exec rm {}\ ;
        - \- *.tmp files are put in {}
        - \- The meaning of "\" is that the command is executed while searching one by one.
        - \- "\+" finds everything and executes the command at once
        
      - $Practice 1$. Find general files whose contents have been changed within the last 24 hours under the current directory and save the list as mtime_b24.txt file
        - find ./ -mtime -1 -type f > mtime_b24.txt
      - $Practice 2$. If it goes beyond the 3rd level under the current directory, it is not searched, and all files that satisfy the condition are copied to the "~/backup" directory.
        - find ./ -maxdepth 3 ... -exec cp {} ~/backup \;

- \- **stdio** (standard input/output)
  - file channel: file에 입출력하기 위한 통로
    - \- file에 channel에 입출력을 하기 위해서 하드웨어에 직접 전근하지 않고, 표준화된 입출력 방식을 통하도록 하는 가상화 레이어의 일종
      - \- 파일채널: 파일에 입출력을 하기 위한 메타 정보를 가지는 객체
    - \- C언어의 I/O 인터페이스의 심플함을 가능하게 함
  - 파일 서술자 (file descriptor, fd로 많이 사용됨)
    - \- 파일 채널들에게는 붙여진 유일한 idenfifier, 숫자로 명명
    - 양수 0번부터 시작하여 증가
    - 예약된 파일 서술자: 0 (stdin), 1 (stdout), 2(stderr) 
fd 값은 프로세스 안에서 부여받는 것이다.
같은 파일을 두번 열수는 없나?
- 같은 프로세스가 같은 파일을 연다고 해도 새로운 pd값이 부여된다. 가능은 하지만 업데이트가 중복될 가능성이 있다.
- PIPE 프로세스 사이에 통신으로 사용
  - IPC(inter-process communication)의 일종
  - pipe의 종류
    - anonymous pipe
      - 프로세스들의 직렬 연결 "(A|B|C)"
      - 임시로 생성되었다가 소멸되는 파이프
      - shell에서 "|"(vertical bar)를 쓰면 생성된다.
      - ex. "find ~ | wc -l"
        - "find ~": 홈디렉토리 이하의 파일들을 모두 찾아달라
        - "|" 파이프가 있으니깐 wc(word count)이므로 line별 이니깐 홈디렉토리 아래에 몇 개의 파일이 있는지 알려달라가 됨.
        - find 명령의 출력(stdout)이 wc 명령의 입력(stdin)과 연결된다.
        - "find ~ > tmp.txt; wc -l < temp.txt; rm tmp.txt"와 동일한 의미를 갖게됨.
        - fd: 1이 fd:0과 pipe로 연결됨
        - wc -l 옵션 사용시 line 수를 카운트한다.
    - named pipe
      - 유닉스에서는 named pipe의 구현체를 FIFO pipe라고 부른다.
      - 파일처럼 구성되어서 path+filename이 있다.
      - path를 가지는 것을 명명되었다고 표현한다.
      - mkfifo or POSIX C API
- Redirection
  - 채널의 방향을 다른 곳으로 연결
  - A > B : A의 stdout을 파일 B로 연결 (저장)
    - ls -a > ls.txt
  - A < B : A의 stdin을 파일 B로 연결
  - A >> B: 방향은 > 과 같고, append mode
    ex. strace ls 2> strace.txt
      - 2>의 의미는 2번 파일서술자를 파일로 연결하는 명령
      - fd 2인 stderr의 출력을 파일로 저장하는 것이다.

- $cat$: stdout와 파일을 자유롭게 연결해주는 기본 필터
  - 파일의 내용을 stdout으로 출력하는 용도
  - stdin의 입력을 redirection해서 파일로 출력하는 용도
  - ex. cat ~/.bashrc
  - ex. cat > hello.txt 그 다음 Hello world를 작성하고 ^D를 하게 되면 Hello world를 hello.txt에 입력하고 종료됨




<center>
  <figure>
    <img src='{{"/assets/images/blog/dev/1_linux/file descripter.png" | relative_url}}' width="70%">
    <figcaption>Figure 3. File descriptor.</figcaption>
  </figure>
</center>


- \- **Change data**
    - \- command
      - \- $cp, mv, rm, mkdir/rmdir, ln$
    - \- $mkdir$: make directory
    - \- $rmdir$: remove directory (In many cases, files and directories are deleted together with $rm -rf$ instead of $rmdir$.)
    - \- $cp$: copy
    - \- $mv$: move, rename
    - \- $rm$: remove

- \- **Meta change**
   - \- command
    - \- $chmod$: change mode 
    - \- $chown, chgrp$: change owner/group

- \- **Archive**
   - \- $tar$
  - archive는 여러 파일을 묶는 작업
  - tar -ctxv
    - c(create) : 아카이브를 생성
    - t(test) : 아카이브를 테스트
    - x(extract) : 아카이브로부터 파일을 풀어냄
    - v(verbose) : 상세한 정보 출력 (실무에서 쓰지 않음)
  - f archive-file : 입출력할 아카이브 파일명 --exclude file: 대상중 file을 제외
  - ex. tar c *.c > arc_c.tar == tar cf arc_c.tar *.c (f옵션을 주면 됨. *.c 파일이 arc_c.tar에 들어감)


- \- **Compress**
   - \- $gzip, zstd$
  - 압축률은 **xz** > bzip2 > **zstd** > gzip > lz4
  - xz: 압축률은 좋으나 느리다.
  - zstd: 요즘 많이 사용

  ex. tar와 gzip을 함께 사용하는 고전적 방법
  - 압축: tar c /etc/*.conf | gzip -c > etc.tar.gz
  - 해제: gzip -cd etc.tar.gz | tar x

#### Text
- \- Editor
   - \- vim(vi)
- \- Filter
   - \- cat(tac), head, tail, less/more, sort
- \- Regex
   - \- grep, sed, awk

#### Job control
  - \- jobs, fg, bg

#### process control
  - \- kill, pkill, pgrep, strace(tracing)

#### Networking
  - \- nc (net cat), curl, wget

#### Disk
  - \- df

#### System
  - \- free, top, ps, pidstat, lshw

### 3.2. Admin commands

#### Package
- \- Redhat: rpm, yum
- \- Debian: dpkg, apt

#### Network
- \- status: ss, netstat(old fashion)
- \- config: nmcli, ip
- \- ssh
- \- packet: tcpdump, wireshark, tshark

#### Files and kernel
- \- lsof
- \- sysctl

#### Disks
- \- fdisk, parted, mkfs, mount, lsblk, blkid, grubby, udisksctl

#### User
- \- useradd, groupadd, usermod
- \- passwd, chpasswd

## 4.