#!/bin/bash

set -e

echo "==================================================================="
echo "Anthropic API 环境变量配置脚本"
echo "注意：本脚本需要在bash环境中运行"
echo "Windows用户请在git bash终端环境下使用"
echo "Mac/Linux用户可直接在终端中运行"
echo "==================================================================="

# 1. 检查终端环境
echo "正在检查运行环境..."

# 检查是否为bash环境
if [ -z "$BASH_VERSION" ]; then
    echo "❌ 错误: 当前不是bash环境"
    echo "请在bash终端中运行此脚本："
    echo "  - Windows: 使用 Git Bash 或 WSL"
    echo "  - Mac/Linux: 使用系统终端"
    exit 1
fi

# 检测操作系统
OS_TYPE="unknown"
case "$(uname -s)" in
    Linux*)     OS_TYPE="Linux";;
    Darwin*)    OS_TYPE="Mac";;
    CYGWIN*|MINGW*|MSYS*) OS_TYPE="Windows";;
    *)          OS_TYPE="unknown";;
esac

echo "✓ 检测到操作系统: $OS_TYPE"
echo "✓ bash环境检查通过 (版本: $BASH_VERSION)"

# Node.js 安装函数
install_nodejs() {
    local platform=$(uname -s)
    
    case "$platform" in
        Linux|Darwin)
            echo "🚀 正在安装 Node.js..."
            
            echo "📥 下载并安装 nvm..."
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
            
            echo "🔄 加载 nvm 环境..."
            \. "$HOME/.nvm/nvm.sh"
            
            echo "📦 下载并安装 Node.js v22..."
            nvm install 22
            
            echo -n "✅ Node.js 安装完成！版本: "
            node -v
            echo -n "✅ npm 版本: "
            npm -v
            ;;
        *)
            echo "❌ 不支持的平台: $platform"
            echo "请手动安装 Node.js: https://nodejs.org/"
            exit 1
            ;;
    esac
}

# 检查 Node.js 环境
echo "检查 Node.js 环境..."
if command -v node >/dev/null 2>&1; then
    current_version=$(node -v | sed 's/v//')
    major_version=$(echo $current_version | cut -d. -f1)
    
    if [ "$major_version" -ge 18 ]; then
        echo "✓ Node.js 已安装: v$current_version"
    else
        echo "⚠️  Node.js v$current_version 版本过低 (需要 >= 18)，正在升级..."
        install_nodejs
    fi
else
    echo "📦 Node.js 未安装，正在安装..."
    install_nodejs
fi

# 检查 npm 环境
if command -v npm >/dev/null 2>&1; then
    echo "✓ npm 已安装: $(npm -v)"
else
    echo "❌ npm 未找到，Node.js 安装可能有问题"
    exit 1
fi

# 2. 确定环境变量配置文件
echo "正在扫描所有可能的环境变量配置文件..."

# 初始化配置文件数组
CONFIG_FILES=()

# 检测当前shell类型
current_shell=$(basename "$SHELL")

# 根据shell类型和操作系统，列出所有可能的配置文件
case "$current_shell" in
    bash)
        # Bash配置文件优先级顺序
        if [ "$OS_TYPE" = "Mac" ]; then
            # macOS上bash配置文件
            potential_files=(
                "$HOME/.bash_profile"
                "$HOME/.bashrc"
                "$HOME/.profile"
            )
        else
            # Linux/Windows上bash配置文件
            potential_files=(
                "$HOME/.bashrc"
                "$HOME/.bash_profile"
                "$HOME/.profile"
            )
        fi
        ;;
    zsh)
        # Zsh配置文件优先级顺序
        potential_files=(
            "$HOME/.zshrc"
            "$HOME/.zprofile"
            "$HOME/.zshenv"
            "$HOME/.profile"
        )
        
        # 检查是否使用Oh My Zsh，避免冲突
        if [ -n "$ZSH" ] && [ -d "$ZSH" ]; then
            echo "⚠️  检测到Oh My Zsh环境，将在配置文件末尾添加变量"
        fi
        ;;
    fish)
        # Fish shell配置文件
        potential_files=(
            "$HOME/.config/fish/config.fish"
        )
        
        # 创建fish配置目录（如果不存在）
        if [ ! -d "$HOME/.config/fish" ]; then
            mkdir -p "$HOME/.config/fish"
            echo "创建fish配置目录: ~/.config/fish/"
        fi
        ;;
    *)
        # 其他shell的通用配置文件
        potential_files=(
            "$HOME/.profile"
            "$HOME/.bashrc"
        )
        ;;
esac

# 检查每个可能的配置文件
echo "检查以下配置文件："
for file in "${potential_files[@]}"; do
    if [ -f "$file" ]; then
        CONFIG_FILES+=("$file")
        echo "  ✓ 找到: ${file/#$HOME/~}"
    else
        echo "  × 不存在: ${file/#$HOME/~}"
    fi
done

# 如果没有找到任何配置文件，创建默认的
if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    # 根据shell类型创建默认配置文件
    case "$current_shell" in
        bash)
            if [ "$OS_TYPE" = "Mac" ]; then
                DEFAULT_FILE="$HOME/.bash_profile"
            else
                DEFAULT_FILE="$HOME/.bashrc"
            fi
            ;;
        zsh)
            DEFAULT_FILE="$HOME/.zshrc"
            ;;
        fish)
            DEFAULT_FILE="$HOME/.config/fish/config.fish"
            ;;
        *)
            DEFAULT_FILE="$HOME/.profile"
            ;;
    esac
    
    touch "$DEFAULT_FILE"
    CONFIG_FILES+=("$DEFAULT_FILE")
    echo "创建新的配置文件: ${DEFAULT_FILE/#$HOME/~}"
fi

echo ""
echo "✓ 将更新 ${#CONFIG_FILES[@]} 个配置文件"

# 3. 检查现有配置（支持不同shell语法）
echo ""
echo "检查现有Anthropic配置..."
EXISTING_CONFIGS=()
BACKUP_FILES=()

# 检查每个配置文件中的现有配置
for config_file in "${CONFIG_FILES[@]}"; do
    has_config=false
    
    # 根据文件名判断语法类型
    if [[ "$config_file" == *"fish"* ]]; then
        # fish shell 语法: set -x ANTHROPIC_AUTH_TOKEN
        if grep -q "set -x ANTHROPIC_AUTH_TOKEN\|set -x ANTHROPIC_BASE_URL" "$config_file" 2>/dev/null; then
            has_config=true
        fi
    else
        # bash/zsh 语法: export ANTHROPIC_AUTH_TOKEN
        if grep -q "ANTHROPIC_AUTH_TOKEN\|ANTHROPIC_BASE_URL" "$config_file" 2>/dev/null; then
            has_config=true
        fi
    fi
    
    if [ "$has_config" = true ]; then
        EXISTING_CONFIGS+=("$config_file")
        echo "⚠️  在 ${config_file/#$HOME/~} 中检测到已存在的Anthropic配置："
        if [[ "$config_file" == *"fish"* ]]; then
            grep -n "set -x ANTHROPIC_" "$config_file" | sed 's/^/     /' || true
        else
            grep -n "ANTHROPIC_" "$config_file" | sed 's/^/     /' || true
        fi
    fi
done

# 如果有现有配置，询问是否覆盖
if [ ${#EXISTING_CONFIGS[@]} -gt 0 ]; then
    echo ""
    echo "📋 在 ${#EXISTING_CONFIGS[@]} 个文件中发现现有配置"
    read -p "是否要覆盖所有现有配置？(y/N): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo "操作已取消"
        exit 0
    fi
    
    # 备份所有包含配置的文件
    echo ""
    echo "正在备份现有配置文件..."
    for config_file in "${EXISTING_CONFIGS[@]}"; do
        backup_file="${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$config_file" "$backup_file"
        BACKUP_FILES+=("$backup_file")
        echo "  ✓ 已备份: ${backup_file/#$HOME/~}"
    done
fi

# 颜色定义
colorReset='\033[0m'
colorBright='\033[1m'
colorCyan='\033[36m'
colorYellow='\033[33m'
colorMagenta='\033[35m'
colorRed='\033[31m'
colorBlue='\033[34m'
colorWhite='\033[37m'
colorGreen='\033[32m'

# 显示API密钥获取横幅
show_api_banner() {
    printf "${colorBright}${colorRed}   █████╗ ██╗  ${colorBlue}██████╗ ██████╗ ██████╗ ███████╗${colorMagenta} ██╗    ██╗██╗████████╗██╗  ██╗${colorReset}\n"
    printf "${colorBright}${colorRed}  ██╔══██╗██║ ${colorBlue}██╔════╝██╔═══██╗██╔══██╗██╔════╝${colorMagenta} ██║    ██║██║╚══██╔══╝██║  ██║${colorReset}\n"
    printf "${colorBright}${colorRed}  ███████║██║ ${colorBlue}██║     ██║   ██║██║  ██║█████╗  ${colorMagenta} ██║ █╗ ██║██║   ██║   ███████║${colorReset}\n"
    printf "${colorBright}${colorRed}  ██╔══██║██║ ${colorBlue}██║     ██║   ██║██║  ██║██╔══╝  ${colorMagenta} ██║███╗██║██║   ██║   ██╔══██║${colorReset}\n"
    printf "${colorBright}${colorRed}  ██║  ██║██║ ${colorBlue}╚██████╗╚██████╔╝██████╔╝███████╗${colorMagenta} ╚███╔███╔╝██║   ██║██╗██║  ██║${colorReset}\n"
    printf "${colorBright}${colorRed}  ╚═╝  ╚═╝╚═╝ ${colorBlue} ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝${colorMagenta}  ╚══╝╚══╝ ╚═╝   ╚═╝╚═╝╚═╝  ╚═╝${colorReset}\n"
    printf "\n"
    printf "${colorBright}${colorYellow}🌐 请从以下网址获取您的API密钥：${colorReset}\n"
    printf "${colorBright}${colorCyan}📋 https://aicodewith.com/dashboard/api-keys${colorReset}\n"
    printf "\n"
    printf "${colorBright}${colorGreen}📝 API密钥格式: sk-acw-********-****************${colorReset}\n"
    printf "\n"
}

# 4. 获取API密钥
echo ""
show_api_banner

# 输入API密钥并验证
while true; do
    read -p "请输入ANTHROPIC_AUTH_TOKEN: " auth_token
    echo ""
    
    # 基本格式验证
    if [[ "$auth_token" =~ ^sk-acw-.{8}-.{16}$ ]]; then
        echo "✓ API密钥格式验证通过"
        break
    else
        echo "❌ API密钥格式不正确"
        echo "   正确格式: sk-acw-********-****************"
        echo "   请重新输入"
    fi
done

# 5. 更新配置文件
echo ""
echo "正在更新配置文件..."
UPDATE_COUNT=0
FAILED_FILES=()

# 处理每个配置文件
for config_file in "${CONFIG_FILES[@]}"; do
    echo "  📝 处理: ${config_file/#$HOME/~}"
    
    # 判断文件类型和语法
    is_fish=false
    if [[ "$config_file" == *"fish"* ]]; then
        is_fish=true
    fi
    
    # 移除旧的Anthropic配置
    if grep -q "ANTHROPIC_AUTH_TOKEN\|ANTHROPIC_BASE_URL" "$config_file" 2>/dev/null || \
       grep -q "set -x ANTHROPIC_AUTH_TOKEN\|set -x ANTHROPIC_BASE_URL" "$config_file" 2>/dev/null; then
        
        # 创建临时文件，移除旧配置
        temp_file=$(mktemp)
        if [ "$is_fish" = true ]; then
            # 移除fish语法的配置行
            grep -v "set -x ANTHROPIC_AUTH_TOKEN\|set -x ANTHROPIC_BASE_URL" "$config_file" > "$temp_file"
        else
            # 移除bash/zsh语法的配置行
            grep -v "ANTHROPIC_AUTH_TOKEN\|ANTHROPIC_BASE_URL" "$config_file" > "$temp_file"
        fi
        mv "$temp_file" "$config_file"
    fi
    
    # 添加新配置
    if [ "$is_fish" = true ]; then
        # fish shell 语法
        {
            echo ""
            echo "# Anthropic API Configuration - $(date '+%Y-%m-%d %H:%M:%S')"
            echo "set -x ANTHROPIC_AUTH_TOKEN $auth_token"
            echo "set -x ANTHROPIC_BASE_URL https://api.jiuwanliguoxue.com/"
        } >> "$config_file"
    else
        # bash/zsh 语法
        {
            echo ""
            echo "# Anthropic API Configuration - $(date '+%Y-%m-%d %H:%M:%S')"
            echo "export ANTHROPIC_AUTH_TOKEN=$auth_token"
            echo "export ANTHROPIC_BASE_URL=https://api.jiuwanliguoxue.com/"
        } >> "$config_file"
    fi
    
    # 验证是否写入成功
    if [ "$is_fish" = true ]; then
        if grep -q "set -x ANTHROPIC_AUTH_TOKEN $auth_token" "$config_file" && \
           grep -q "set -x ANTHROPIC_BASE_URL" "$config_file"; then
            echo "     ✓ 配置成功写入"
            ((UPDATE_COUNT++))
        else
            echo "     ❌ 配置写入失败"
            FAILED_FILES+=("$config_file")
        fi
    else
        if grep -q "ANTHROPIC_AUTH_TOKEN=$auth_token" "$config_file" && \
           grep -q "ANTHROPIC_BASE_URL=" "$config_file"; then
            echo "     ✓ 配置成功写入"
            ((UPDATE_COUNT++))
        else
            echo "     ❌ 配置写入失败"
            FAILED_FILES+=("$config_file")
        fi
    fi
done

echo ""
echo "✓ 成功更新 $UPDATE_COUNT/${#CONFIG_FILES[@]} 个配置文件"

# 如果有失败的文件，显示错误信息
if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo ""
    echo "❌ 以下文件更新失败："
    for failed_file in "${FAILED_FILES[@]}"; do
        echo "   - ${failed_file/#$HOME/~}"
    done
fi

# 6. 加载环境变量并验证
echo ""
echo "正在加载和验证环境变量..."

# 尝试从非fish配置文件加载环境变量
if [[ "$current_shell" != "fish" ]]; then
    # 从所有非fish配置文件中提取并加载Anthropic环境变量
    for config_file in "${CONFIG_FILES[@]}"; do
        if [[ "$config_file" != *"fish"* ]]; then
            eval "$(grep "^export ANTHROPIC_" "$config_file" 2>/dev/null || true)"
        fi
    done
else
    echo "⚠️  Fish shell配置文件不兼容bash，跳过自动加载"
fi

# 手动设置环境变量用于当前会话
export ANTHROPIC_AUTH_TOKEN=$auth_token
export ANTHROPIC_BASE_URL=https://api.jiuwanliguoxue.com/

# 验证配置是否成功
if [ "$UPDATE_COUNT" -eq "${#CONFIG_FILES[@]}" ]; then
    echo "✅ 所有配置文件更新成功！"
    echo ""
    echo "📊 当前配置:"
    echo "   ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
    echo "   ANTHROPIC_AUTH_TOKEN: ${ANTHROPIC_AUTH_TOKEN:0:12}...(已隐藏)"
    echo ""
    
    # 显示更新的配置文件列表
    echo "📋 已更新的配置文件："
    for config_file in "${CONFIG_FILES[@]}"; do
        echo "   - ${config_file/#$HOME/~}"
    done
    echo ""
    echo "🎉 配置完成！"
    echo ""
    
    # 7. 检查并安装/更新Claude Code客户端
    echo "🔍 检查Claude Code客户端..."
    if command -v claude >/dev/null 2>&1; then
        echo "✓ Claude Code已安装: $(claude --version)"
        echo ""
        echo "🚀 是否要更新Claude Code客户端到最新版本？"
        read -p "这将执行: npm uninstall/install -g @anthropic-ai/claude-code (y/N): " update_claude
        
        if [[ "$update_claude" =~ ^[Yy]$ ]]; then
            echo "🔄 正在更新Claude Code客户端..."
            
            echo "步骤1: 卸载旧版本..."
            npm uninstall -g @anthropic-ai/claude-code
            
            echo "步骤2: 安装最新版本..."
            if npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com; then
                echo "✅ Claude Code客户端更新成功！"
            else
                echo "❌ Claude Code客户端安装失败，请手动执行："
                echo "   npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com"
            fi
        fi
    else
        echo "📦 Claude Code未安装，正在安装..."
        if npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com; then
            echo "✅ Claude Code客户端安装成功！"
        else
            echo "❌ Claude Code客户端安装失败，请手动执行："
            echo "   npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com"
            exit 1
        fi
    fi
    
    # 8. 配置Claude Code跳过引导
    echo ""
    echo "🔧 配置Claude Code跳过引导..."
    node --eval "
        const fs = require('fs');
        const os = require('os');
        const path = require('path');
        
        const homeDir = os.homedir(); 
        const filePath = path.join(homeDir, '.claude.json');
        
        try {
            if (fs.existsSync(filePath)) {
                const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
                fs.writeFileSync(filePath, JSON.stringify({ ...content, hasCompletedOnboarding: true }, null, 2), 'utf-8');
                console.log('✅ 已更新现有Claude配置文件');
            } else {
                fs.writeFileSync(filePath, JSON.stringify({ hasCompletedOnboarding: true }, null, 2), 'utf-8');
                console.log('✅ 已创建Claude配置文件并跳过引导');
            }
        } catch (error) {
            console.log('⚠️  配置Claude引导跳过时出错:', error.message);
        }
    "
    echo ""
    
    # 9. 检测并清理Claude配置文件中的代理设置
    echo ""
    echo "🔍 检测Claude配置文件中的代理设置..."
    # Claude配置文件可能的路径（优先检查settings.json）
    CLAUDE_SETTING_FILE=""
    if [ -f "$HOME/.claude/settings.json" ]; then
        CLAUDE_SETTING_FILE="$HOME/.claude/settings.json"
    elif [ -f "$HOME/.claude/settings.local.json" ]; then
        CLAUDE_SETTING_FILE="$HOME/.claude/settings.local.json"
    elif [ -f "$HOME/.claude/setting.json" ]; then
        CLAUDE_SETTING_FILE="$HOME/.claude/setting.json"
    fi
    
    if [ -n "$CLAUDE_SETTING_FILE" ]; then
        echo "✓ 找到Claude配置文件: ${CLAUDE_SETTING_FILE/#$HOME/~}"
        
        # 检测是否存在代理设置
        PROXY_FOUND=false
        PROXY_SETTINGS=""
        
        # 检查是否有HTTP代理设置（不区分大小写）
        if grep -iq "http_proxy\|https_proxy\|httpproxy\|httpsproxy" "$CLAUDE_SETTING_FILE" 2>/dev/null; then
            PROXY_FOUND=true
            echo ""
            echo "⚠️  检测到残留的代理配置："
            PROXY_SETTINGS=$(grep -in "http_proxy\|https_proxy\|httpproxy\|httpsproxy" "$CLAUDE_SETTING_FILE" | sed 's/^/   /')
            echo "$PROXY_SETTINGS"
            echo ""
            echo "📝 这些代理设置可能会影响Claude Code的正常使用"
            echo "   建议删除这些设置以避免连接问题"
            echo ""
            
            read -p "是否要删除这些代理设置？(y/N): " remove_proxy
            if [[ "$remove_proxy" =~ ^[Yy]$ ]]; then
                # 备份原配置文件
                backup_claude_file="${CLAUDE_SETTING_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
                cp "$CLAUDE_SETTING_FILE" "$backup_claude_file"
                echo "✓ 已备份Claude配置到: ${backup_claude_file/#$HOME/~}"
                
                # 删除代理设置行（不区分大小写）
                # 使用sed删除包含代理相关设置的行
                if [[ "$OS_TYPE" = "Mac" ]]; then
                    # Mac版本的sed需要备份文件参数
                    sed -i '' '/[Hh][Tt][Tt][Pp]_[Pp][Rr][Oo][Xx][Yy]\|[Hh][Tt][Tt][Pp][Ss]_[Pp][Rr][Oo][Xx][Yy]\|[Hh][Tt][Tt][Pp][Pp][Rr][Oo][Xx][Yy]\|[Hh][Tt][Tt][Pp][Ss][Pp][Rr][Oo][Xx][Yy]/d' "$CLAUDE_SETTING_FILE"
                else
                    # Linux/Windows版本的sed
                    sed -i '/[Hh][Tt][Tt][Pp]_[Pp][Rr][Oo][Xx][Yy]\|[Hh][Tt][Tt][Pp][Ss]_[Pp][Rr][Oo][Xx][Yy]\|[Hh][Tt][Tt][Pp][Pp][Rr][Oo][Xx][Yy]\|[Hh][Tt][Tt][Pp][Ss][Pp][Rr][Oo][Xx][Yy]/d' "$CLAUDE_SETTING_FILE"
                fi
                
                # 验证删除结果（不区分大小写）
                if ! grep -iq "http_proxy\|https_proxy\|httpproxy\|httpsproxy" "$CLAUDE_SETTING_FILE" 2>/dev/null; then
                    echo "✅ 代理设置已成功删除"
                    echo "📋 Claude Code现在应该能正常使用默认网络连接"
                else
                    echo "❌ 代理设置删除失败"
                    echo "   请手动编辑文件: $CLAUDE_SETTING_FILE"
                    echo "   或恢复备份: cp $backup_claude_file $CLAUDE_SETTING_FILE"
                fi
            else
                echo "跳过代理设置清理"
            fi
        else
            echo "✓ 未发现代理设置，配置文件正常"
        fi
    else
        echo "ℹ️  未找到Claude配置文件（${CLAUDE_SETTING_FILE/#$HOME/~}）"
        echo "   这是正常的，配置文件会在首次使用Claude Code时自动创建"
    fi
    echo ""
    
# 显示配置完成横幅
show_complete_banner() {
    printf "\n"
    printf "${colorBright}${colorRed}   █████╗ ██╗  ${colorBlue}██████╗ ██████╗ ██████╗ ███████╗${colorMagenta} ██╗    ██╗██╗████████╗██╗  ██╗${colorReset}\n"
    printf "${colorBright}${colorRed}  ██╔══██╗██║ ${colorBlue}██╔════╝██╔═══██╗██╔══██╗██╔════╝${colorMagenta} ██║    ██║██║╚══██╔══╝██║  ██║${colorReset}\n"
    printf "${colorBright}${colorRed}  ███████║██║ ${colorBlue}██║     ██║   ██║██║  ██║█████╗  ${colorMagenta} ██║ █╗ ██║██║   ██║   ███████║${colorReset}\n"
    printf "${colorBright}${colorRed}  ██╔══██║██║ ${colorBlue}██║     ██║   ██║██║  ██║██╔══╝  ${colorMagenta} ██║███╗██║██║   ██║   ██╔══██║${colorReset}\n"
    printf "${colorBright}${colorRed}  ██║  ██║██║ ${colorBlue}╚██████╗╚██████╔╝██████╔╝███████╗${colorMagenta} ╚███╔███╔╝██║   ██║██╗██║  ██║${colorReset}\n"
    printf "${colorBright}${colorRed}  ╚═╝  ╚═╝╚═╝ ${colorBlue} ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝${colorMagenta}  ╚══╝╚══╝ ╚═╝   ╚═╝╚═╝╚═╝  ╚═╝${colorReset}\n"
    printf "\n"
    printf "${colorBright}${colorYellow}📌 请执行以下命令使配置立即生效：${colorReset}\n"
    printf "${colorBright}${colorCyan}   source ${CONFIG_FILE/#$HOME/~}${colorReset}\n"
    printf "\n"
    printf "${colorBright}${colorGreen}🔄 或者重启终端让配置自动生效${colorReset}\n"
    printf "\n"
}

    show_complete_banner
    echo ""
    echo "🔧 如需修改配置，可编辑: ${CONFIG_FILE/#$HOME/~}"
else
    # 方案3: 改进错误提示，说明可能的原因
    echo "❌ 配置文件验证失败，可能的原因："
    echo "   1. 配置文件写入过程中出现错误"
    echo "   2. 磁盘空间不足或权限问题"
    echo "   3. API密钥格式在写入时被意外修改"
    echo ""
    echo "🔍 调试信息："
    echo "   配置文件路径: $CONFIG_FILE"
    echo "   API密钥长度: ${#auth_token}"
    echo "   配置文件末尾内容:"
    tail -5 "$CONFIG_FILE" 2>/dev/null || echo "   无法读取配置文件"
    echo ""
    echo "💡 建议解决方案："
    echo "   1. 检查磁盘空间: df -h $HOME"
    echo "   2. 检查文件权限: ls -la $CONFIG_FILE"
    echo "   3. 手动验证配置: cat $CONFIG_FILE | grep ANTHROPIC"
    echo "   4. 重新运行脚本"
    exit 1
fi